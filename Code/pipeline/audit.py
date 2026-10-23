"""Consistency check of the stored results; reads only the CSV tables of each run."""
import ast
from collections import defaultdict
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from . import benchmark, csv_results as T, experiment, matching_runner

RUNGS = ('all', '1', '2', '3', '4')
RATE_FIELDS = ('total', 'assigned', 'correct', 'wrong_assigned', 'unassigned', 'accuracy', 'coverage',
               'precision_assigned')


def matching_jobs(c, recipe):
    values = {k: v for k, v in vars(c).items() if k.isupper()}
    values.update(recipe['matching'])
    values.update(DURCHLAEUFE=1)
    return matching_runner.build_jobs(SimpleNamespace(**values), {'settings': recipe['benchmark']})


def result_table(c, number):
    with open(Path(c.ERGEBNIS_ORDNER) / f'run_{number:02d}.csv', newline='') as stream:
        return list(csv.DictReader(stream))


def _float(text):
    return None if text in ('', None) else float(text)


def _same_number(text, value):
    """Compares a value of run_XX.csv (written with 15 significant digits) with a recomputed value."""
    if value is None or text in ('', None):
        return value is None and text in ('', None)
    return text == (format(value, '.15g') if isinstance(value, float) else str(value)) or float(text) == float(value)


def check_predictions(number, s, evaluation, rows):
    by_id = {r['report_id']: r for r in evaluation}
    methods = benchmark.method_tasks(s)
    expected = {(m, t, rid) for m, tasks in methods.items() for t in tasks for rid in by_id}
    keys = [(r['method'], r['task'], r['report_id']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f'run_{number:02d}: predictions missing, duplicated or unexpected.')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['task'], r['method'], 'all')].append(r['correct'])
        groups[(r['task'], r['method'], str(by_id[r['report_id']]['difficulty_level']))].append(r['correct'])
    return groups


def check_matching(c, number, recipe, evaluation, rows, events):
    by_id = {r['report_id']: r for r in evaluation}
    jobs = matching_jobs(c, recipe)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['source_method'], r['pass3_method'])].append(r)
    planned = {(j['source_method'], j['pass3_method']) for j in jobs}
    if not planned <= set(grouped):
        raise ValueError(f'run_{number:02d}: matching.csv lacks matching jobs of Config.py.')
    scores = {}
    for key, group in grouped.items():
        if sorted(r['report_id'] for r in group) != sorted(by_id):
            raise ValueError(f'run_{number:02d}/{key[0]}: matching decisions do not cover the evaluation reports.')
        assigned = [r['matched_event_id'] for r in group if r['matched_event_id'] is not None]
        if len(assigned) != len(set(assigned)):
            raise ValueError(f'run_{number:02d}/{key[0]}: event matched more than once.')
        for r in group:
            truth = by_id[r['report_id']]['ground_truth']
            target = events[r['target_event_id']]
            reference = {'Activity': truth['Activity'], 'Resource': truth['Machine'],
                         'End Time': truth['Error time and Error date'], 'Worker ID': truth['WorkerID']}
            if any(r[f'truth_{k}'] != reference[k] or str(target[k]) != reference[k] for k in T.EVENT_FIELDS):
                raise ValueError(f'run_{number:02d}: reference event differs from the ground truth ({r["report_id"]}).')
            event = events.get(r['matched_event_id']) if r['matched_event_id'] is not None else None
            correct = bool(event) and all(str(event[k]) == reference[k] for k in T.EVENT_FIELDS)
            if correct != r['correct']:
                raise ValueError(f'run_{number:02d}: matching correctness differs ({r["report_id"]}).')
        for rung in RUNGS:
            part = group if rung == 'all' else [r for r in group if str(r['difficulty_level']) == rung]
            scores[(key[0], key[1], rung)] = T.rates(part)
    return scores


def check_training(number, s, rows):
    """All baselines of a run are trained with one seed; it is taken from training.csv."""
    seeds = {r['method']: r.get('seed') for r in rows}
    fits = [c['fit_method'] for c in benchmark.learning_curve.conditions(s) if c['fit_method'] is not None]
    if any(fit not in seeds for fit in fits):
        raise ValueError(f'run_{number:02d}: training.csv lacks trained baselines.')
    used = {seeds[fit] for fit in fits}
    if len(used) != 1 or not isinstance(next(iter(used)), int):
        raise ValueError(f'run_{number:02d}: baselines were not trained with one seed.')
    return used.pop()


def check_results(number, table, prediction_groups, matching_scores):
    seen = set()
    for row in table:
        task, method, rung = row['task'], row['method'], row['rung']
        if task in ('classification', 'extraction'):
            values = prediction_groups.get((task, method, rung))
            if values is None or int(row['n_scored']) != len(values) or not _same_number(row['accuracy'], sum(values) / len(values)):
                raise ValueError(f'run_{number:02d}: run_{number:02d}.csv differs from predictions.csv ({task}, {method}, {rung}).')
            seen.add((task, method, rung))
        elif task == 'matching':
            scores = matching_scores.get((method, row['pass3_method'], rung))
            if scores is None or any(not _same_number(row[k], scores[k]) for k in RATE_FIELDS):
                raise ValueError(f'run_{number:02d}: run_{number:02d}.csv differs from matching.csv ({method}, {rung}).')
            seen.add(('matching', method, row['pass3_method'], rung))
    if len(seen) != len(prediction_groups) + len(matching_scores):
        raise ValueError(f'run_{number:02d}: run_{number:02d}.csv does not cover all predictions and matching jobs.')


def check_run(c, number, events=None):
    recipe = experiment.recipe_for(c, number)
    s = experiment.settings_from_recipe(c, recipe)
    benchmark.validate(s)
    corpus, _ = benchmark.load_corpus(s)
    evaluation = benchmark.split_rows(corpus, s, 0)['eval']
    if events is None:
        with c.EVENTLOG_DATEI.open(newline='') as stream:
            events = {r['original_index']: r for r in csv.DictReader(stream)}
    predictions = T.read_predictions(c, number)
    groups = check_predictions(number, s, evaluation, predictions)
    scores = check_matching(c, number, recipe, evaluation, T.read_matching(c, number), events)
    seed = check_training(number, s, T.read_training(c, number))
    table = result_table(c, number)
    check_results(number, table, groups, scores)
    if not any(r['source'] == 'benchmark' for r in T.read_models(c, number)):
        raise ValueError(f'run_{number:02d}: models.csv is empty.')
    return {'predictions': len(predictions), 'matching_jobs': len({k[:2] for k in scores}), 'result_rows': len(table),
            'seed': seed,
            'identity': {(r['task'], r['method'], r['rung'], r.get('pass3_method') or '') for r in table}}


def check_files(c):
    count = 0
    for path in c.PAKET.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise ValueError(f'Package contains a file symlink: {path}')
        count += 1
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == '.json':
            json.loads(path.read_text())
        elif path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                if line.strip():
                    json.loads(line)
        elif path.suffix == '.csv':
            with path.open(newline='', encoding='utf-8') as stream:
                reader = csv.reader(stream)
                header = next(reader, [])
                if len(header) != len(set(header)):
                    raise ValueError(f'Duplicate CSV columns: {path}')
                if any(len(row) != len(header) for row in reader):
                    raise ValueError(f'Inconsistent CSV width: {path}')
    return count


def check(c):
    numbers = T.finished_runs(c)
    if not numbers:
        raise ValueError('No result runs found.')
    files = check_files(c)
    with c.EVENTLOG_DATEI.open(newline='') as stream:
        events = {r['original_index']: r for r in csv.DictReader(stream)}
    common, total = None, defaultdict(int)
    for number in numbers:
        result = check_run(c, number, events=events)
        identity = result.pop('identity')
        seed = result.pop('seed')
        if common is not None and identity != common:
            if not (common <= identity or identity <= common):
                raise ValueError(f'run_{number:02d}: experiment conditions differ across runs.')
        common = identity if common is None else common & identity
        for k, v in result.items():
            total[k] += v
        print(f"run_{number:02d}: {result['predictions']} predictions, {result['result_rows']} result rows, "
              f"{result['matching_jobs']} matching jobs, baseline seed {seed}.", flush=True)
    unfinished = experiment.run_directories(c)
    if unfinished:
        print('Unfinished runs in the working folder: ' + ', '.join(p.name for p in unfinished), flush=True)
    summary = {'files': files, 'runs': len(numbers), **total}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
