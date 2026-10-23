from __future__ import annotations

import csv
from decimal import ROUND_HALF_UP, Decimal
import json
from pathlib import Path

import pandas as pd

LLMS = (
    ('llama3_2_1b_false', 'Llama 3.2 1B', 'Tiny'),
    ('llama3_2_3b_false', 'Llama 3.2 3B', 'Tiny'),
    ('gemma3_4b_false', 'Gemma 3 4B', 'Tiny'),
    ('ornith-1_5_9b_false', 'Ornith 1.5 9B', 'Small'),
    ('gemma3_12b_false', 'Gemma 3 12B', 'Small'),
    ('gemma4_e4b_false', 'Gemma 4 E4B', 'Small'),
    ('gpt-oss_20b_low', 'GPT-OSS 20B (low)', 'Small'),
    ('gpt-oss_20b_high', 'GPT-OSS 20B (high)', 'Small'),
    ('gemma3_27b_false', 'Gemma 3 27B', 'Medium'),
    ('gemma4_31b_false', 'Gemma 4 31B (off)', 'Medium'),
    ('gemma4_31b_true', 'Gemma 4 31B (on)', 'Medium'),
    ('nemotron3_33b_false', 'Nemotron 3 Nano Omni 33B', 'Medium'),
    ('llama3_1_70b_false', 'Llama 3.1 70B', 'Large'),
    ('gpt-oss_120b_medium', 'GPT-OSS 120B (medium)', 'Large'),
    ('nemotron-3-super_120b_false', 'Nemotron 3 Super 120B', 'Large'),
)
GROUPS = ('Tiny', 'Small', 'Medium', 'Large')
MODEL_ROWS = (
    ('llama3.2:1b', 'Llama 3.2 1B', 'Tiny', '--'),
    ('llama3.2:3b', 'Llama 3.2 3B', 'Tiny', '--'),
    ('gemma3:4b', 'Gemma 3 4B', 'Tiny', '--'),
    ('ornith-1.5:9b', 'Ornith 1.5 9B', 'Small', 'Off'),
    ('gemma3:12b', 'Gemma 3 12B', 'Small', '--'),
    ('gemma4:e4b', 'Gemma 4 E4B', 'Small', 'Off'),
    ('gpt-oss:20b', 'GPT-OSS 20B', 'Small', 'Low, high'),
    ('gemma3:27b', 'Gemma 3 27B', 'Medium', '--'),
    ('gemma4:31b', 'Gemma 4 31B', 'Medium', 'Off, on'),
    ('nemotron3:33b', 'Nemotron 3 Nano Omni 33B', 'Medium', 'Off'),
    ('llama3.1:70b', 'Llama 3.1 70B', 'Large', '--'),
    ('gpt-oss:120b', 'GPT-OSS 120B', 'Large', 'Medium'),
    ('nemotron-3-super:120b', 'Nemotron 3 Super 120B', 'Large', 'Off'),
)
MEMORY_GB = {'llama3.2:1b': 1.32, 'llama3.2:3b': 2.02, 'gemma3:4b': 3.34, 'ornith-1.5:9b': 6.55,
             'gemma3:12b': 8.15, 'gemma4:e4b': 9.61, 'gpt-oss:20b': 13.79, 'gemma3:27b': 17.40,
             'gemma4:31b': 19.87, 'nemotron3:33b': 27.64, 'llama3.1:70b': 42.52, 'gpt-oss:120b': 65.37,
             'nemotron-3-super:120b': 86.83}
NAMES = {key: name for key, name, _ in LLMS}
GROUP = {key: group for key, _, group in LLMS}
SHOTS = (0, 1, 3)
LEVELS = ('1', '2', '3', '4')
CLASSIFICATION_BASELINES = (('tfidf_logreg', 'TF-IDF + LR'), ('tfidf_svm', 'TF-IDF + SVM'), ('bert', 'BERT'))
EXTRACTION_BASELINES = (('crf_time', 'CRF + Time'), ('ner_time', 'DistilBERT-NER + Time'),
                        ('bert_ner_time', 'BERT-NER + Time'))
TRAINING = ((10, 60), (25, 150), (50, 300), (75, 450), (100, 600))
BASELINE_MATCHER = 'llm_gemma3_12b_false'


def llm_method(key, shots):
    return f'llm_{key}_{shots}shot'


def baseline_method(name, percent):
    return name if percent == 100 else f'{name}__train_{percent:03d}'


class Results:
    """Reads only the CSV tables: Results/run_XX.csv and Results/run_XX/*.csv."""

    def __init__(self, folder):
        from types import SimpleNamespace
        from pipeline import csv_results
        self.folder = Path(folder)
        self._tables = csv_results
        self._c = SimpleNamespace(ERGEBNIS_ORDNER=self.folder)
        self.runs = [self.folder / f'run_{n:02d}' for n in csv_results.finished_runs(self._c)]
        frames = [pd.read_csv(self.folder / f'{p.name}.csv', dtype={'rung': str}) for p in self.runs]
        self.rows = pd.concat(frames, ignore_index=True)
        self.rows['pass3_method'] = self.rows['pass3_method'].fillna('')
        self._predictions, self._matching = {}, {}

    def values(self, task, method, column='accuracy', rung='all', pass3='', runs=None):
        rows = self.rows
        rows = rows[(rows.task == task) & (rows.method == method) & (rows.rung == rung) & (rows.pass3_method == pass3)]
        if runs is not None:
            rows = rows[rows.run.isin(runs)]
        if len(rows) == 0:
            raise KeyError(f'No results for {task}/{method}/{rung}/{pass3}')
        return rows[column]

    def mean(self, *args, **kwargs):
        return float(self.values(*args, **kwargs).mean())

    def percent(self, *args, **kwargs):
        return 100 * self.mean(*args, **kwargs)

    def best_shots(self, task, key):
        return max(SHOTS, key=lambda s: (round(self.mean(task, llm_method(key, s)), 12), -s))

    def best_matching_shots(self, key):
        return max(SHOTS, key=lambda s: (round(self.matching(llm_method(key, s)), 12), -s))

    def matching_pass3(self, method):
        return f'llm_{method.rsplit("_", 1)[0][4:]}' if method.startswith('llm_') else BASELINE_MATCHER

    def matching(self, method, column='accuracy', rung='all', runs=None):
        return self.mean('matching', method, column, rung, self.matching_pass3(method), runs)

    PREDICTION_COLUMNS = ('task', 'method', 'report_id', 'difficulty_level', 'true_category', 'prediction',
                          'correct', 'included_in_benchmark')
    MATCHING_COLUMNS = ('source_method', 'pass3_method', 'report_id', 'matched_event_id', 'match_method', 'correct',
                        *[f'truth_{k}' for k in ('Activity', 'Resource', 'End Time', 'Worker ID')])

    def _prediction_rows(self, run):
        if run not in self._predictions:
            rows = self._tables.read_predictions(self._c, run, keep=self.PREDICTION_COLUMNS)
            self._predictions[run] = [r for r in rows if r['included_in_benchmark']]
        return self._predictions[run]

    def _matching_rows(self, run):
        if run not in self._matching:
            jobs = {}
            for r in self._tables.read_matching(self._c, run, keep=self.MATCHING_COLUMNS):
                jobs.setdefault((r['source_method'], r['pass3_method']), []).append(r)
            self._matching[run] = jobs
        return self._matching[run]

    def predictions(self, run, methods, tasks=('classification', 'extraction')):
        return {(r['task'], r['method'], r['report_id']): r for r in self._prediction_rows(run)
                if r['method'] in methods and r['task'] in tasks}

    def _job(self, run, method):
        return self._matching_rows(run).get((method, self.matching_pass3(method)), [])

    def decisions(self, run, method):
        return [dict(r, truth={k: r[f'truth_{k}'] for k in EVENT_FIELDS}) for r in self._job(run, method)]

    def models(self):
        rows = self._tables.read_models(self._c, self.run_numbers()[0])
        return {r['model']: {'digest': r['digest'], 'details': {'family': r['family'], 'format': r['format'],
                'parameter_size': r['parameter_size'], 'quantization_level': r['quantization_level']}}
                for r in rows if r['source'] == 'benchmark'}

    def run_numbers(self):
        return [int(p.name[4:]) for p in self.runs]


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_events(path):
    with open(path, newline='') as stream:
        return {row['original_index']: row for row in csv.DictReader(stream)}


EVENT_FIELDS = ('Activity', 'Resource', 'End Time', 'Worker ID')


def matched_correctly(decision, events):
    event = events.get(decision['matched_event_id']) if decision['matched_event_id'] is not None else None
    return bool(event) and all(str(event[k]) == str(decision['truth'][k]) for k in EVENT_FIELDS)


def fmt(value, digits=1):
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(round(float(value), 9))).quantize(quantum, rounding=ROUND_HALF_UP))
