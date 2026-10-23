from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
from statistics import mean, stdev

from .data import (CLASSIFICATION_BASELINES, EXTRACTION_BASELINES, LLMS, NAMES, GROUP, TRAINING, llm_method)

LEVEL1 = ('single_candidate',)
LEVEL2 = ('cosine_direct', 'attribute_direct')
LEVEL3 = ('llm_choice', 'abstained', 'invalid_response', 'no_unused_candidates')
PROMPTS = {0: 'zero-shot', 1: 'one-shot', 3: 'three-shot'}
BASELINE_NAMES = {'regex': 'Regular expressions', 'crf': 'CRF', 'crf_time': 'CRF + Time', 'ner': 'DistilBERT-NER',
                  'ner_time': 'DistilBERT-NER + Time', 'bert_ner': 'BERT-NER', 'bert_ner_time': 'BERT-NER + Time',
                  **dict(CLASSIFICATION_BASELINES)}
REPORTS = dict(TRAINING)
COLUMNS_FIXED = ['component', 'method', 'model', 'size_group', 'prompt', 'best_prompt', 'training_reports',
                 'difficulty', 'metric']
TASK_METRICS = {
    'classification': [('accuracy_pct', 'accuracy', 100), ('macro_f1_pct', 'macro_f1', 100)],
    'extraction': [('accuracy_pct', 'accuracy', 100), ('activity_accuracy_pct', 'accuracy_Activity', 100),
                   ('machine_accuracy_pct', 'accuracy_Machine', 100), ('worker_accuracy_pct', 'accuracy_WorkerID', 100),
                   ('time_accuracy_pct', 'accuracy_Error time and Error date', 100)],
}
MATCHING_METRICS = [('accuracy_pct', 'accuracy', 100), ('correct', 'correct', 1), ('assigned', 'assigned', 1),
                    ('wrong_assigned', 'wrong_assigned', 1), ('unassigned', 'unassigned', 1),
                    ('correct_assignments_pct', 'precision_assigned', 100)]


def describe(method):
    if method.startswith('llm_'):
        key, shots = method.rsplit('_', 1)
        key, shots = key[4:], int(shots.removesuffix('shot'))
        return {'model': NAMES[key], 'size_group': GROUP[key], 'prompt': PROMPTS[shots], 'key': key, 'shots': shots}
    name, _, percent = method.partition('__train_')
    return {'model': BASELINE_NAMES[name], 'size_group': 'NLP baselines', 'prompt': '', 'key': name, 'shots': None,
            'training_reports': REPORTS[int(percent) if percent else 100]}


class Summary:
    def __init__(self, results):
        self.r = results
        self.runs = results.run_numbers()
        self.rows = []
        self.table = results.rows
        self._correct = {}

    def add(self, base, metric, values):
        values = {run: values.get(run) for run in self.runs}
        present = [v for v in values.values() if v is not None]
        if not present:
            return
        row = dict(base, metric=metric, **{f'run_{run:02d}': values[run] for run in self.runs})
        row['mean'] = mean(present)
        row['sd'] = stdev(present) if len(present) > 1 else None
        self.rows.append(row)

    def base(self, component, method, difficulty='all', best=None):
        info = describe(method)
        return {'component': component, 'method': method, 'model': info['model'], 'size_group': info['size_group'],
                'prompt': info['prompt'], 'best_prompt': '' if best is None else ('yes' if best else 'no'),
                'training_reports': info.get('training_reports', ''), 'difficulty': difficulty}

    def methods(self, task):
        present = set(self.table[self.table.task == task].method)
        llms = [llm_method(key, s) for key, _, _ in LLMS for s in PROMPTS if llm_method(key, s) in present]
        order = list(BASELINE_NAMES)
        baselines = sorted((m for m in present if not m.startswith('llm_')),
                           key=lambda m: (order.index(m.split('__train_')[0]), describe(m)['training_reports']))
        return llms + baselines

    def value(self, task, method, column, rung='all', pass3=''):
        rows = self.table[(self.table.task == task) & (self.table.method == method) & (self.table.rung == rung)
                          & (self.table.pass3_method == pass3)]
        result = {}
        for _, row in rows.iterrows():
            value = row[column]
            result[int(row.run)] = None if value != value else float(value)
        return result

    def best(self, task, method):
        info = describe(method)
        if info['shots'] is None:
            return None
        if task == 'matching':
            return info['shots'] == self.r.best_matching_shots(info['key'])
        return info['shots'] == self.r.best_shots(task, info['key'])

    def predictions(self, run):
        return self.r._prediction_rows(run)

    def task_rows(self, task):
        counts = defaultdict(dict)
        if task == 'classification':
            for run in self.runs:
                per_method = defaultdict(Counter)
                for p in self.predictions(run):
                    if p['task'] == 'classification':
                        per_method[p['method']][p['prediction']] += 1
                for method, counter in per_method.items():
                    counts[method][run] = counter
        labels = sorted({label for m in counts.values() for c in m.values() for label in c})
        for method in self.methods(task):
            best = self.best(task, method)
            for rung in ('all', '1', '2', '3', '4'):
                base = self.base(task, method, rung, best)
                for metric, column, factor in TASK_METRICS[task]:
                    values = {run: None if v is None else v * factor
                              for run, v in self.value(task, method, column, rung).items()}
                    self.add(base, metric, values)
            for label in labels:
                self.add(self.base(task, method, 'all', best), f'predicted_{label}',
                         {run: counts[method][run][label] for run in counts[method]})

    def matching_methods(self):
        rows = self.table[self.table.task == 'matching']
        methods = []
        for method in self.methods('matching'):
            pass3 = self.r.matching_pass3(method)
            runs = set(rows[(rows.method == method) & (rows.pass3_method == pass3)].run)
            if runs == set(self.runs):
                methods.append(method)
        return methods

    def correct(self, run, task, method):
        if (run, task) not in self._correct:
            maps = defaultdict(dict)
            for p in self.predictions(run):
                if p['task'] == task:
                    maps[p['method']][p['report_id']] = p['correct']
            self._correct[(run, task)] = maps
        return self._correct[(run, task)][method]

    def matching_rows(self):
        for method in self.matching_methods():
            pass3 = self.r.matching_pass3(method)
            best = self.best('matching', method)
            for rung in ('all', '1', '2', '3', '4'):
                base = self.base('matching', method, rung, best)
                for metric, column, factor in MATCHING_METRICS:
                    values = {run: None if v is None else v * factor
                              for run, v in self.value('matching', method, column, rung, pass3).items()}
                    self.add(base, metric, values)
            levels = defaultdict(dict)
            for run in self.runs:
                decisions = self.r._job(run, method)
                extracted = self.correct(run, 'extraction', method)
                assigned = [d for d in decisions if d['matched_event_id'] is not None]
                failed = [d for d in decisions if not extracted.get(d['report_id'], False)]
                level2 = sum(d['correct'] and d['match_method'] in LEVEL2 for d in failed)
                level3 = sum(d['correct'] and d['match_method'] in LEVEL3 for d in failed)
                values = {
                    'level1_assigned': sum(d['match_method'] in LEVEL1 for d in assigned),
                    'level2_assigned': sum(d['match_method'] in LEVEL2 for d in assigned),
                    'level3_reached': sum(d['match_method'] in LEVEL3 for d in decisions),
                    'level3_assigned': sum(d['match_method'] in LEVEL3 for d in assigned),
                    'level3_correct': sum(d['correct'] and d['match_method'] in LEVEL3 for d in decisions),
                    'incorrect_extractions': len(failed),
                    'incorrect_extractions_matched_level2': level2,
                    'incorrect_extractions_matched_level3': level3,
                    'incorrect_extractions_matched_pct': 100 * (level2 + level3) / len(failed) if failed else None,
                }
                for metric, value in values.items():
                    levels[metric][run] = value
            base = self.base('matching', method, 'all', best)
            for metric, values in levels.items():
                self.add(base, metric, values)

    def pipeline_rows(self):
        cls_methods = [llm_method(k, self.r.best_shots('classification', k)) for k, _, _ in LLMS]
        cls_methods += [k for k, _ in CLASSIFICATION_BASELINES]
        ext_methods = [llm_method(k, self.r.best_matching_shots(k)) for k, _, _ in LLMS]
        ext_methods += [k for k, _ in EXTRACTION_BASELINES]
        for cls in cls_methods:
            for ext in ext_methods:
                values = {}
                for run in self.runs:
                    correct = self.correct(run, 'classification', cls)
                    decisions = self.r._job(run, ext)
                    values[run] = 100 * mean(bool(correct[d['report_id']]) and d['correct'] for d in decisions)
                a, b = describe(cls), describe(ext)
                base = {'component': 'pipeline', 'method': f'{cls} + {ext}', 'model': f"{a['model']} + {b['model']}",
                        'size_group': '', 'prompt': '', 'best_prompt': '', 'training_reports': '', 'difficulty': 'all'}
                self.add(base, 'accuracy_pct', values)

    def runtime_rows(self):
        for key, _, _ in LLMS:
            cls = llm_method(key, self.r.best_shots('classification', key))
            ext = llm_method(key, self.r.best_matching_shots(key))
            pass3 = self.r.matching_pass3(ext)
            parts = {
                'classification_sec_per_report': self.value('classification', cls, 'mean_wall_time_sec'),
                'extraction_sec_per_report': self.value('extraction', ext, 'mean_wall_time_sec'),
            }
            component = self.value('matching', ext, 'component_wall_sec', pass3=pass3)
            total = self.value('matching', ext, 'total', pass3=pass3)
            parts['matching_sec_per_report'] = {run: component[run] / total[run] for run in component}
            parts['total_sec_per_report'] = {run: sum(p[run] for p in parts.values()) for run in self.runs}
            base = dict(self.base('runtime', ext, 'all'), prompt='', best_prompt='')
            base['method'] = f'{cls} / {ext}'
            for metric, values in parts.items():
                self.add(base, metric, values)

    def build(self):
        self.task_rows('classification')
        self.task_rows('extraction')
        self.matching_rows()
        self.pipeline_rows()
        self.runtime_rows()
        return self.rows


def number(value):
    if value is None:
        return ''
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f'{value:.4f}'.rstrip('0').rstrip('.')
    return str(value)


def write(results, path):
    summary = Summary(results)
    rows = summary.build()
    columns = COLUMNS_FIXED + [f'run_{run:02d}' for run in summary.runs] + ['mean', 'sd']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(columns)
        for row in rows:
            writer.writerow([number(row.get(c)) if c.startswith('run_') or c in ('mean', 'sd') else row.get(c, '')
                             for c in columns])
    return path, len(rows)
