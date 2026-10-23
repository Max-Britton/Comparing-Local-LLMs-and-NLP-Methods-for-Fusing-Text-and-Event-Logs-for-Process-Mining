from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import re
from types import SimpleNamespace

from . import benchmark, matching_runner

MATCHING_KEYS = (
    'MATCHING_QUELLEN', 'MATCHING_TRAININGSANTEILE', 'MATCHING_QUELL_DURCHLAUF',
    'MATCHING_LLM_MODUS', 'BASELINE_MATCHING_MODELL', 'MATCHING_MODELLE',
    'MATCHING_REIHENFOLGE_MISCHEN', 'MATCHING_REIHENFOLGE_SEED',
    'EMBEDDING_MODELL', 'VERGLEICH', 'KOSINUS_SCHWELLE', 'MINDESTABSTAND', 'TOP_K',
    'MIN_ATTRIBUTE_DIREKT', 'NORMALISIERUNG_PASS1', 'ZEITFENSTER_SEKUNDEN',
    'MIT_ORIGINALBERICHT', 'HYBRID_SCHWELLE', 'GEWICHT_AKTIVITAET', 'GEWICHT_MASCHINE',
    'GEWICHT_DATUM', 'GEWICHT_UHRZEIT', 'GEWICHT_WORKER', 'ZEIT_HALBWERT_MINUTEN',
    'MODELL_SEED', 'TEMPERATUR', 'KONTEXT_TOKEN', 'ANTWORT_TOKEN_MATCHING',
    'ANTWORT_TOKEN_MIT_THINKING', 'BASELINE_TRAININGSANTEILE', 'MODELLE',
    'EXTRAKTIONS_BASELINES', 'DURCHLAEUFE',
)


def load(path):
    return json.loads(Path(path).read_text())


def run_directories(c):
    paths = [p for p in c.DETAIL_ORDNER.glob('run_*') if p.is_dir() and re.fullmatch(r'run_\d{2,}', p.name)]
    return sorted(paths, key=lambda p: int(p.name[4:]))


def next_number(c):
    from .csv_results import finished_runs
    numbers = [int(p.name[4:]) for p in run_directories(c)] + finished_runs(c)
    return max(numbers, default=0) + 1


def run_complete(c, folder):
    if not benchmark_complete(folder):
        return False
    context = matching_context(c, folder)
    return not matching_runner.pending_jobs(context, matching_runner.build_jobs(context, load(folder / 'benchmark/config.json')))


def new_settings(c, number):
    settings = benchmark.Settings(
        input_path=c.TEXT_DATEI, output=c.DETAIL_ORDNER / f'run_{number:02d}' / 'benchmark',
        models=c.MODELLE, tasks=['extraction', 'classification'],
        extraction_baselines=c.EXTRAKTIONS_BASELINES,
        classification_baselines=c.KLASSIFIKATIONS_BASELINES,
        extraction_shots=c.EXTRAKTION_SHOTS, classification_shots=c.KLASSIFIKATION_SHOTS,
        training_percentages=c.BASELINE_TRAININGSANTEILE, progress=c.FORTSCHRITT_ANZEIGEN,
        repetitions=1, fractions=(.4, .1, .5), resplit=False, split_path=c.SPLIT_DATEI,
        split_seed=c.SPLIT_SEED, training_subset_seed=c.TRAININGS_SEED_START + number - 1,
        validation_subset_seed=c.VALIDIERUNGS_SEED,
        model_seed=c.BASELINE_SEED_START + number - 1, llm_seed=c.MODELL_SEED,
        host=c.OLLAMA_HOST, temperature=c.TEMPERATUR, num_ctx=c.KONTEXT_TOKEN,
        extraction_tokens=c.ANTWORT_TOKEN_EXTRAKTION,
        classification_tokens=c.ANTWORT_TOKEN_KLASSIFIKATION,
        thinking_tokens=c.ANTWORT_TOKEN_MIT_THINKING, timeout=c.TIMEOUT_SEKUNDEN,
        ner_model=str(c.NER_MODELL), ner_epochs=c.NER_EPOCHEN, ner_batch_size=c.NER_BATCH_GROESSE,
        ner_lr=c.NER_LERNRATE, ner_device=c.NER_GERAET, bert_model=str(c.BERT_MODELL),
        bert_device=c.NER_GERAET, compact_outputs=True, tune_baselines=True, validation_proportional=True)
    seeds = {condition['fit_method']: settings.model_seed
             for condition in benchmark.learning_curve.conditions(settings)
             if condition['fit_method'] is not None}
    return replace(settings, method_seeds=seeds)


def recipe(c, number, settings):
    return {'run': number, 'benchmark': benchmark.portable_settings(settings),
            'matching': {key: getattr(c, key) for key in MATCHING_KEYS}}


def recipe_for(c, number):
    """Settings of a run as defined by Config.py."""
    return recipe(c, number, new_settings(c, number))


def same_recipe(a, b):
    """Compares two run settings, ignoring the Ollama address and the location of the working data."""
    import json
    ignore = {'host', 'output'}
    strip = lambda r: json.loads(json.dumps({**r, 'benchmark': {k: v for k, v in r['benchmark'].items() if k not in ignore}}, default=str))
    return strip(a) == strip(b)


def settings_from_recipe(c, value):
    values = dict(value['benchmark'])
    for key in ('input_path', 'output', 'split_path', 'ner_model', 'bert_model'):
        if values.get(key) is not None:
            path = Path(values[key])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Non-portable path in {key}: {path}')
            resolved = (c.PAKET / path).resolve()
            if not resolved.is_relative_to(c.PAKET.resolve()):
                raise ValueError(f'Path leaves package: {path}')
            values[key] = str(resolved) if key.endswith('_model') else resolved
    values.update(host=c.OLLAMA_HOST, progress=c.FORTSCHRITT_ANZEIGEN)
    return benchmark.Settings(**values)


def matching_context(c, folder):
    value = load(folder / 'run.json')
    values = {k: v for k, v in vars(c).items() if k.isupper()}
    values.update(value['matching'])
    values.update(ERGEBNIS_ORDNER=folder, MATCHING_BENCHMARK_ORDNER=folder / 'benchmark',
                  MATCHING_BENCHMARK_LISTE=None, DURCHLAEUFE=1)
    return SimpleNamespace(**values)


def benchmark_complete(folder):
    path = folder / 'benchmark/metrics.json'
    if not path.exists():
        return False
    rows = load(path)['results']
    return bool(rows) and all(r['complete'] for r in rows)


def plan_matching(c, folders=None, announce=True, llm_only=False):
    entries = []
    for folder in run_directories(c) if folders is None else folders:
        context = matching_context(c, folder)
        if not benchmark_complete(folder):
            raise ValueError(f'{folder.name}: finish the benchmark before matching.')
        jobs = matching_runner.build_jobs(context, load(folder / 'benchmark/config.json'))
        if llm_only:
            jobs = [job for job in jobs if job['source_method'].startswith('llm_')]
        pending = matching_runner.pending_jobs(context, jobs)
        entries.append((folder, context, jobs, pending))
        if announce:
            print(f'{folder.name}: {len(jobs)-len(pending)}/{len(jobs)} fertig; {len(pending)} offen.', flush=True)
    if announce:
        print(f'{sum(len(e[3]) for e in entries)} offene Matching-Kombinationen.', flush=True)
    return entries


@contextmanager
def series_lock(c):
    c.ERGEBNIS_ORDNER.mkdir(parents=True, exist_ok=True)
    yield


def run_matching(c, folders=None, plan_only=False, client_factory=None, llm_only=False):
    if plan_only:
        return plan_matching(c, folders, llm_only=llm_only)
    with series_lock(c):
        from .export import export_run
        for folder, context, jobs, pending in plan_matching(c, folders, llm_only=llm_only):
            if pending:
                matching_runner.run(context, jobs=pending, client_factory=client_factory,
                                    on_progress=lambda: export_run(c, folder))
            export_run(c, folder)
            if run_complete(c, folder):
                from .csv_results import finalize
                finalize(c, folder)


def select_run(c, requested=None):
    active = c.ERGEBNIS_ORDNER / 'active_run.json'
    if requested is not None:
        if type(requested) is not int or requested < 1:
            raise ValueError('Run number must be a positive integer.')
        from .csv_results import finished_runs
        if requested in finished_runs(c) and not (c.DETAIL_ORDNER / f'run_{requested:02d}').exists():
            raise ValueError(f'run_{requested:02d} is finished and stored as CSV; start a new run instead.')
        return requested
    return load(active)['run'] if active.exists() else next_number(c)


def run(c, requested=None, plan_only=False, client_factory=None):
    number = select_run(c, requested)
    folder = c.DETAIL_ORDNER / f'run_{number:02d}'
    value = load(folder / 'run.json') if (folder / 'run.json').exists() else recipe(c, number, new_settings(c, number))
    s = settings_from_recipe(c, value)
    benchmark.validate(s)
    if plan_only:
        print(f'Vollständiger Durchlauf: run_{number:02d}', flush=True)
        print(f'{len(benchmark.llm_conditions(s))} LLM-Bedingungen; {len(benchmark.learning_curve.conditions(s))} Baseline-Bedingungen.', flush=True)
        print('Extraktion → Klassifikation → Matching (Baselines: 100 %) → CSV-Export. Keine Dateien oder Modellaufrufe.', flush=True)
        return number
    with series_lock(c):
        if requested is None and select_run(c) != number:
            raise RuntimeError('Run allocation changed; start again.')
        folder.mkdir(parents=True, exist_ok=True)
        benchmark.freeze(folder / 'run.json', benchmark.payload(value))
        active = c.ERGEBNIS_ORDNER / 'active_run.json'
        benchmark.atomic(active, benchmark.payload({'run': number}))
        if not benchmark_complete(folder):
            if client_factory is None:
                from ollama import Client
                factory = Client
            else:
                factory = client_factory
            client = factory(host=c.OLLAMA_HOST, timeout=c.TIMEOUT_SEKUNDEN)
            context = matching_context(c, folder)
            jobs = matching_runner.build_jobs(context, {'settings': value['benchmark']})
            selected = {benchmark.method_id(m): m for m in s.models}
            selected.update({job['pass3_method']: job['pass3_model'] for job in jobs})
            benchmark.preflight(client, list(selected.values()))
            matching_runner.model_digest(list(client.list().models), c.EMBEDDING_MODELL)
            benchmark.run(s, client_factory=client_factory)
        from .export import export_run
        if benchmark_complete(folder):
            from .retention import compact_training
            compact_training(folder / 'benchmark')
        export_run(c, folder)
        if not benchmark_complete(folder):
            raise RuntimeError('Benchmark is incomplete; restart to resume it.')
        context = matching_context(c, folder)
        matching_runner.run(context, client_factory=client_factory,
                            on_progress=lambda: export_run(c, folder))
        export_run(c, folder)
        if run_complete(c, folder):
            from .csv_results import finalize
            finalize(c, folder)
            active.unlink(missing_ok=True)
    return folder
