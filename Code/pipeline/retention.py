import json
from pathlib import Path
import shutil


def compact_training(output):
    from .benchmark import atomic, payload
    records = Path(output) / 'records'
    keep = {'fit_config.json', 'training.json', 'training_history.json', 'selection_config.json',
            'selection_trials.json', 'train_annotations.json', 'validation_annotations.json'}
    for folder in records.iterdir():
        if not folder.is_dir() or folder.name in ('training_subsets', 'validation_subsets'):
            continue
        for path in folder.iterdir():
            if path.name not in keep:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            elif path.suffix == '.json':
                data = json.loads(path.read_text())
                def clean(value):
                    if isinstance(value, dict):
                        return {k: clean(v) for k, v in value.items() if not k.endswith('_sha256')}
                    if isinstance(value, list):
                        return [clean(v) for v in value]
                    return value
                atomic(path, payload(clean(data)))
    (records / 'progress.json').unlink(missing_ok=True)
    (Path(output) / 'prompts.jsonl').unlink(missing_ok=True)
    shutil.rmtree(Path(output) / 'ollama', ignore_errors=True)


MATCHING_REMOVED = (
    'input_events.csv',
    'selected_predictions.jsonl',
    'matching_inputs.jsonl',
    'report_metadata.jsonl',
    'matched_reports.csv',
    'validation_results.csv',
    'matched_eventlog.csv',
    'timing/requests.jsonl',
)


def compact_matching(output):
    for path in Path(output).glob('*/model_lifecycle.json'):
        path.unlink()
    for job in Path(output).iterdir():
        if job.is_dir() and (job / 'completed.json').is_file():
            for name in MATCHING_REMOVED:
                (job / name).unlink(missing_ok=True)
