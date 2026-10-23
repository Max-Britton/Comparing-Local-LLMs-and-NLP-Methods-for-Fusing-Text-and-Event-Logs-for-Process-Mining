from __future__ import annotations

from statistics import mean

from .data import (CLASSIFICATION_BASELINES, EXTRACTION_BASELINES, GROUPS, LEVELS, LLMS, NAMES, SHOTS,
                   TRAINING, baseline_method, llm_method, matched_correctly)


def classification_table(r):
    llms = [{'key': key, 'name': name, 'group': group,
             'values': {s: (r.percent('classification', llm_method(key, s)),
                            r.percent('classification', llm_method(key, s), 'macro_f1')) for s in SHOTS}}
            for key, name, group in LLMS]
    baselines = [{'key': key, 'name': name,
                  'values': {n: (r.percent('classification', baseline_method(key, p)),
                                 r.percent('classification', baseline_method(key, p), 'macro_f1')) for p, n in TRAINING}}
                 for key, name in CLASSIFICATION_BASELINES]
    return llms, baselines


def extraction_table(r):
    llms = []
    for key, name, group in LLMS:
        best = r.best_shots('extraction', key)
        method = llm_method(key, best)
        llms.append({'key': key, 'name': name, 'group': group, 'best_shots': best,
                     'overall': {s: r.percent('extraction', llm_method(key, s)) for s in SHOTS},
                     'levels': {l: r.percent('extraction', method, rung=l) for l in LEVELS},
                     'best': r.percent('extraction', method)})
    baselines = [{'key': key, 'name': name,
                  'training': {n: r.percent('extraction', baseline_method(key, p)) for p, n in TRAINING},
                  'levels': {l: r.percent('extraction', key, rung=l) for l in LEVELS},
                  'best': r.percent('extraction', key)} for key, name in EXTRACTION_BASELINES]
    return llms, baselines


def matching_figure(r):
    rows = []
    for key, name, group in LLMS:
        method = llm_method(key, r.best_matching_shots(key))
        rows.append({'name': name, 'group': group, 'method': method, 'overall': 100 * r.matching(method),
                     'levels': {l: 100 * r.matching(method, rung=l) for l in LEVELS}})
    for key, name in EXTRACTION_BASELINES:
        rows.append({'name': name, 'group': 'NLP baselines', 'method': key, 'overall': 100 * r.matching(key),
                     'levels': {l: 100 * r.matching(key, rung=l) for l in LEVELS}})
    return rows


def runtime_figure(r, matching_runs=None):
    rows = []
    for key, name, group in LLMS:
        cls = llm_method(key, r.best_shots('classification', key))
        ext = llm_method(key, r.best_matching_shots(key))
        total = r.matching(ext, 'total', runs=matching_runs)
        rows.append({'name': name, 'group': group,
                     'classification': r.mean('classification', cls, 'mean_wall_time_sec'),
                     'extraction': r.mean('extraction', ext, 'mean_wall_time_sec'),
                     'matching': r.matching(ext, 'component_wall_sec', runs=matching_runs) / total})
    for row in rows:
        row['total'] = row['classification'] + row['extraction'] + row['matching']
    return rows


def report_level(r, events, methods_cls, methods_ext):
    runs = {}
    for run in r.run_numbers():
        predictions = r.predictions(run, set(methods_cls) | set(methods_ext))
        runs[run] = {
            'cls': {m: {rid: rec['correct'] for (task, mm, rid), rec in predictions.items()
                        if task == 'classification' and mm == m} for m in methods_cls},
            'ext': {m: {rid: rec['correct'] for (task, mm, rid), rec in predictions.items()
                        if task == 'extraction' and mm == m} for m in methods_ext},
            'match': {m: {d['report_id']: (d, matched_correctly(d, events)) for d in r.decisions(run, m)}
                      for m in methods_ext},
        }
    return runs


def pipeline_accuracy(runs, cls_method, ext_method):
    values = []
    for data in runs.values():
        cls, match = data['cls'][cls_method], data['match'][ext_method]
        values.append(100 * mean(bool(cls[rid]) and match[rid][1] for rid in match))
    return mean(values)


def pipeline_table(r, runs):
    best_cls = {key: llm_method(key, r.best_shots('classification', key)) for key, _, _ in LLMS}
    best_ext = {key: llm_method(key, r.best_matching_shots(key)) for key, _, _ in LLMS}
    rows = []
    for index, group in enumerate(GROUPS):
        allowed = [key for key, _, g in LLMS if GROUPS.index(g) <= index]
        options = [(round(pipeline_accuracy(runs, best_cls[c], best_ext[e]), 9), c == e, c, e)
                   for c in allowed for e in allowed]
        value, _, c, e = max(options)
        rows.append({'group': group, 'classification': NAMES[c], 'extraction': NAMES[e],
                     'cls': r.percent('classification', best_cls[c]), 'match': 100 * r.matching(best_ext[e]),
                     'pipeline': value})
    best_baseline = max((k for k, _ in CLASSIFICATION_BASELINES), key=lambda k: r.mean('classification', k))
    options = [(pipeline_accuracy(runs, best_baseline, best_ext[e]), e) for e, _, _ in LLMS]
    value, e = max(options)
    rows.append({'group': 'Combination', 'classification': dict(CLASSIFICATION_BASELINES)[best_baseline],
                 'extraction': NAMES[e], 'cls': r.percent('classification', best_baseline),
                 'match': 100 * r.matching(best_ext[e]), 'pipeline': value})
    return rows, best_cls, best_ext
