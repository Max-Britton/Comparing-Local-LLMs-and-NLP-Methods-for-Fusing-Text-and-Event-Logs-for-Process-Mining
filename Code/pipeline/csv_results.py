"""Per-run result tables (CSV only).

After a run is complete, the working data in Results/Arbeitsdaten/run_XX is condensed into
Results/run_XX.csv (one row per method, task, prompt and difficulty level) and the tables
in Results/run_XX/:

- predictions.csv  every classification and extraction prediction, incl. the raw model answer
- matching.csv     every matching decision with the reference event
- models.csv       Ollama models (digest, size, quantization)
- training.csv     training settings and selected epoch/hyperparameters of the baselines

The settings of each run are those in Config.py; the seed of the trained baselines is stored in training.csv.

All later steps (MainCheck.py, MainPaperResults.py) read only these CSV files.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import re
import shutil

csv.field_size_limit(1 << 30)

FIELD_KEYS = (('Activity', 'Activity'), ('Machine', 'Machine'), ('WorkerID', 'WorkerID'),
              ('Time', 'Error time and Error date'))
OPTION_FIELDS = ('temperature', 'seed', 'num_ctx', 'num_predict')
PREDICTIONS = [
    ('task', 'str'), ('method', 'str'), ('report_id', 'str'), ('difficulty_level', 'int'), ('true_category', 'str'),
    ('model', 'str'), ('thinking', 'thinking'), ('shots', 'int'),
    ('baseline', 'str'), ('training_percent', 'int'), ('training_n', 'int'), ('prediction_partition', 'str'),
    ('temperature', 'float'), ('seed', 'int'), ('num_ctx', 'int'), ('num_predict', 'int'),
    ('raw_output', 'text'), ('raw_baseline_output', 'str'), ('temporal_resolution', 'json'),
    ('prediction', 'prediction'), ('parse_ok', 'bool'), ('correct', 'bool'),
    *[(f'correct_{short}', 'bool') for short, _ in FIELD_KEYS],
    ('validation_failures', 'json'), ('parse_error', 'str'), ('api_error', 'text'), ('included_in_benchmark', 'bool'),
    ('done_reason', 'str'), ('prompt_tokens', 'int'), ('completion_tokens', 'int'), ('answer_chars', 'int'),
    ('thinking_chars', 'int'), ('latency_sec', 'float'), ('wall_time_sec', 'float'), ('ollama_total_sec', 'float'),
    ('ollama_load_sec', 'float'), ('ollama_prompt_eval_sec', 'float'), ('ollama_eval_sec', 'float'),
    ('component_wall_sec', 'float'), ('request_started_at', 'str'), ('request_finished_at', 'str'),
]
EVENT_FIELDS = ('Activity', 'Resource', 'End Time', 'Worker ID')
MATCHING = [
    ('source_method', 'str'), ('pass3_method', 'str'), ('source_shots', 'int'), ('source_training_percent', 'int'),
    ('report_id', 'str'), ('difficulty_level', 'int'), ('target_event_id', 'str'),
    *[(f'truth_{k}', 'str') for k in EVENT_FIELDS],
    ('matched_event_id', 'str'), ('correct', 'bool'),
    ('match_method', 'str'), ('parse_status', 'str'), ('llm_status', 'str'),
    ('filter_mode', 'str'), ('agreeing_attributes', 'int'), ('provided_attributes', 'int'),
    ('filter_candidate_ids', 'json'), ('pass1_normalized', 'bool'), ('pass1_filter_mode', 'str'),
    ('pass1_candidate_ids', 'json'), ('score_mode', 'str'), ('top10_candidate_ids', 'json'), ('top10_scores', 'json'),
    ('top10_attribute_scores', 'json'), ('llm_candidate_ids', 'json'), ('llm_attribute_scores', 'json'),
    ('match_latency_sec', 'float'), ('llm_latency_sec', 'float'), ('match_prompt_tokens', 'int'),
    ('match_completion_tokens', 'int'), ('match_total_tokens', 'int'),
]
MODELS = [('source', 'str'), ('model', 'str'), ('digest', 'str'), ('family', 'str'), ('format', 'str'),
          ('parameter_size', 'str'), ('quantization_level', 'str'), ('parameter_count', 'int'),
          ('capabilities', 'json')]
TRAINING_SKIP = {'train_ids', 'validation_ids', 'history', 'training_diagnostics', 'selection', 'annotation_policy',
                 'setup_policy', 'unresolved_policy', 'normalization'}


def _encode(kind, value):
    if value is None:
        return ''
    if kind == 'bool':
        return 'true' if value else 'false'
    if kind == 'thinking':
        return 'true' if value is True else 'false' if value is False else str(value)
    if kind == 'json':
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if kind == 'prediction':
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if kind == 'float':
        return repr(float(value))
    return str(value)


def _decode(kind, text, row):
    if kind == 'text':
        return text
    if text == '':
        return None
    if kind == 'int':
        return int(text)
    if kind == 'float':
        return float(text)
    if kind == 'bool':
        return {'true': True, 'false': False}[text]
    if kind == 'thinking':
        return {'true': True, 'false': False}.get(text, text)
    if kind == 'json':
        return json.loads(text)
    if kind == 'prediction':
        return json.loads(text) if row.get('task') == 'extraction' else text
    return text


def write_table(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in columns]
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(names)
    for row in rows:
        unknown = set(row) - set(names)
        if unknown:
            raise ValueError(f'{path.name}: unexpected fields {sorted(unknown)}')
        writer.writerow([_encode(kind, row.get(name)) for name, kind in columns])
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(stream.getvalue(), encoding='utf-8')
    os.replace(tmp, path)
    return path


def read_table(path, columns=None, keep=None):
    """Reads a table; without column kinds every cell is read as JSON. keep: only these columns."""
    with open(path, newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if columns is not None and header != [n for n, _ in columns]:
            raise ValueError(f'{path}: unexpected columns.')
        kinds = dict(columns) if columns is not None else {name: 'json' for name in header}
        names = header if keep is None else [n for n in header if n in set(keep)]
        rows = []
        for values in reader:
            raw = dict(zip(header, values))
            rows.append({name: _decode(kinds[name], raw[name], raw) for name in names})
        return rows


def run_folder(c, number):
    return Path(c.ERGEBNIS_ORDNER) / f'run_{number:02d}'


def finished_runs(c):
    """Numbers of runs with Results/run_XX.csv and the tables in Results/run_XX/."""
    numbers = []
    for path in Path(c.ERGEBNIS_ORDNER).glob('run_*.csv'):
        if re.fullmatch(r'run_\d{2,}\.csv', path.name) and (path.with_suffix('') / 'predictions.csv').exists():
            numbers.append(int(path.stem[4:]))
    return sorted(numbers)


def read_predictions(c, number, keep=None):
    return read_table(run_folder(c, number) / 'predictions.csv', PREDICTIONS, keep)


def read_matching(c, number, keep=None):
    return read_table(run_folder(c, number) / 'matching.csv', MATCHING, keep)


def read_models(c, number):
    return read_table(run_folder(c, number) / 'models.csv', MODELS)


def read_training(c, number):
    return read_table(run_folder(c, number) / 'training.csv')


def rates(rows):
    total = len(rows)
    assigned = sum(r['matched_event_id'] is not None for r in rows)
    correct = sum(r['correct'] for r in rows)
    return {'total': total, 'assigned': assigned, 'unassigned': total - assigned, 'correct': correct,
            'wrong_assigned': assigned - correct, 'accuracy': correct / total if total else None,
            'coverage': assigned / total if total else None,
            'precision_assigned': correct / assigned if assigned else None}


def _jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prediction_row(record):
    names = {name for name, _ in PREDICTIONS}
    row = {k: v for k, v in record.items() if k in names}
    options = record.get('options') or {}
    row.update({k: options.get(k) for k in OPTION_FIELDS})
    row['done_reason'] = record.get('ollama_done_reason')
    field = record.get('field_correct') or {}
    row.update({f'correct_{short}': field.get(key) for short, key in FIELD_KEYS})
    return row


def matching_rows(c, folder):
    from . import benchmark, matching_runner
    from .experiment import load, matching_context
    with open(c.EVENTLOG_DATEI, newline='') as stream:
        events = {r['original_index']: r for r in csv.DictReader(stream)}
    evaluation = {r['report_id']: r for r in benchmark.read_rows(folder / 'benchmark/records/eval.jsonl')}
    context = matching_context(c, folder)
    rows = []
    for job in matching_runner.build_jobs(context, load(folder / 'benchmark/config.json')):
        root = folder / 'matching' / job['relative_output']
        truth = {r['report_id']: r for r in _jsonl(root / 'validation_truth.jsonl')}
        for decision in _jsonl(root / 'matching_decisions.jsonl'):
            target = truth[decision['report_id']]
            event = events.get(decision['matched_event_id']) if decision['matched_event_id'] is not None else None
            row = dict(decision)
            row.update(source_method=job['source_method'], pass3_method=job['pass3_method'],
                       source_shots=job.get('source_shots'), source_training_percent=job.get('source_training_percent'),
                       difficulty_level=evaluation[decision['report_id']]['difficulty_level'],
                       target_event_id=target['target_event_id'],
                       correct=bool(event) and all(str(event[k]) == str(target[k]) for k in EVENT_FIELDS))
            row.update({f'truth_{k}': target[k] for k in EVENT_FIELDS})
            rows.append(row)
    return rows


def model_rows(folder):
    from .experiment import load
    rows = []
    for source, path in (('benchmark', folder / 'benchmark/models.json'), ('matching', folder / 'matching/models.json')):
        if not path.exists():
            continue
        for name, info in load(path).items():
            if not isinstance(info, dict):
                continue
            details = info.get('details') or {}
            rows.append({'source': source, 'model': name, 'digest': info.get('digest'),
                         'family': details.get('family'), 'format': details.get('format'),
                         'parameter_size': details.get('parameter_size'),
                         'quantization_level': details.get('quantization_level'),
                         'parameter_count': (info.get('modelinfo') or {}).get('general.parameter_count'),
                         'capabilities': info.get('capabilities')})
    return rows


def training_rows(folder):
    from .experiment import load
    rows = []
    for path in sorted((folder / 'benchmark/records').glob('*/training.json')):
        row = {'method': path.parent.name}
        for prefix, source in (('', path), ('fit_', path.with_name('fit_config.json'))):
            if source.exists():
                for key, value in load(source).items():
                    if key not in TRAINING_SKIP and not (isinstance(value, (dict, list)) and len(json.dumps(value)) > 4000):
                        row[prefix + key] = value
        rows.append(row)
    return rows


def export_components(c, folder):
    """Writes Results/run_XX/*.csv for one run from its working folder."""
    from . import benchmark
    from .prediction_records import completed
    folder = Path(folder)
    out = run_folder(c, int(folder.name[4:]))
    records = completed(benchmark.read_rows(folder / 'benchmark/records/predictions.jsonl'))
    write_table(out / 'predictions.csv', PREDICTIONS, [prediction_row(r) for r in records])
    write_table(out / 'matching.csv', MATCHING, matching_rows(c, folder))
    write_table(out / 'models.csv', MODELS, model_rows(folder))
    training = training_rows(folder)
    columns = ['method'] + sorted({k for r in training for k in r} - {'method'})
    write_table(out / 'training.csv', [(k, 'json') for k in columns], training)
    return out


def _same(kind, a, b):
    if kind == 'str' and b == '':
        b = None
    if kind == 'float' and a is not None and b is not None:
        return float(a) == float(b)
    return a == b


def verify_export(c, folder):
    """Reads the tables back and compares every value with the working data."""
    from . import benchmark
    from .experiment import load
    from .prediction_records import completed
    folder = Path(folder)
    number = int(folder.name[4:])
    stored = completed(benchmark.read_rows(folder / 'benchmark/records/predictions.jsonl'))
    checks = [('predictions', PREDICTIONS, read_predictions(c, number), [prediction_row(r) for r in stored]),
              ('matching', MATCHING, read_matching(c, number), matching_rows(c, folder)),
              ('models', MODELS, read_models(c, number), model_rows(folder))]
    for name, columns, rows, expected in checks:
        if len(rows) != len(expected):
            raise ValueError(f'{folder.name}: {name}.csv has {len(rows)} rows, expected {len(expected)}.')
        for a, b in zip(rows, expected):
            for column, kind in columns:
                if not _same(kind, a.get(column), b.get(column)):
                    raise ValueError(f'{folder.name}: {name}.csv differs in {column} ({a.get("report_id", a.get("model"))}).')
    stored_training = read_training(c, number)
    header = list(stored_training[0]) if stored_training else []
    expected_training = [{k: r.get(k) for k in header} for r in training_rows(folder)]
    if json.loads(json.dumps(stored_training)) != json.loads(json.dumps(expected_training)):
        raise ValueError(f'{folder.name}: training.csv differs from the working data.')
    from .experiment import recipe_for, same_recipe
    if not same_recipe(load(folder / 'run.json'), recipe_for(c, number)):
        raise ValueError(f'{folder.name}: the settings of this run differ from Config.py; '
                         'working data kept. Restore the settings of this run in Config.py and start MainResults.py again.')
    return True


def finalize(c, folder):
    """Exports a complete run to CSV, verifies the tables and removes the working data."""
    from .audit import check_run
    from .export import export_run
    folder = Path(folder)
    number = int(folder.name[4:])
    export_run(c, folder)
    export_components(c, folder)
    verify_export(c, folder)
    check_run(c, number)
    if not getattr(c, 'ARBEITSDATEN_BEHALTEN', False):
        shutil.rmtree(folder)
        if folder.parent.exists() and not any(folder.parent.iterdir()):
            folder.parent.rmdir()
    print(f'{folder.name}: CSV tables in {run_folder(c, number)}.', flush=True)
    return run_folder(c, number)
