from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import time

import pandas as pd

from . import lower_levels, rendering, symptoms, upper_levels, v11

FIELDS = ['original_index', 'Case ID', 'Activity', 'Resource', 'Worker ID', 'End Time']
ERROR_CLASSES = v11.ERROR_CLASSES
MODEL_DIGEST = v11.MISTRAL_DIGEST


def read_events(path):
    frame = pd.read_csv(path, dtype={'Case ID': str, 'Worker ID': str, 'Resource': str})
    if set(FIELDS) - set(frame) or frame[FIELDS].isna().any().any():
        raise ValueError('The event file needs complete columns: ' + ', '.join(FIELDS))
    indexes = pd.to_numeric(frame.original_index, errors='raise')
    if not (indexes % 1 == 0).all() or indexes.duplicated().any():
        raise ValueError('original_index must contain unique integers.')
    frame['original_index'] = indexes.astype('int64')
    if not frame['Worker ID'].str.fullmatch(r'ID\d+').all() or not frame.Resource.str.fullmatch(r'Machine \d+').all():
        raise ValueError('Expected formats: Machine 5 and ID0042.')
    pd.to_datetime(frame['End Time'], format='%Y/%m/%d %H:%M', errors='raise')
    return frame


def event_key(row):
    truth = row['ground_truth']
    return (str(row['case_id']), truth['Activity'], truth['Machine'], truth['WorkerID'], truth['Error time and Error date'])


def choose_least(counts, values, rng):
    minimum = min(counts[v] for v in values)
    chosen = rng.choice([v for v in values if counts[v] == minimum])
    counts[chosen] += 1
    return chosen


def make_sources(frame, existing, count, seed):
    styles = symptoms.backup_templates()
    used_indexes = {r['original_index'] for r in existing}
    used_keys = {event_key(r) for r in existing}
    unique = frame.drop_duplicates(subset=FIELDS[1:])
    candidates = [row for row in unique.to_dict('records')
                  if row['original_index'] not in used_indexes
                  and tuple(str(row[f]) for f in FIELDS[1:]) not in used_keys]
    if count > len(candidates):
        raise ValueError(f'{count} new events are needed; {len(candidates)} are available.')
    offset = len(existing)
    random.Random(seed + 10000 + offset).shuffle(candidates)
    label_rng, level_rng = random.Random(seed + 20000 + offset), random.Random(seed + 25000 + offset)
    class_counts = Counter(r['true_category'] for r in existing)
    level_counts = Counter(r['difficulty_level'] for r in existing)
    sources = []
    for raw in candidates[:count]:
        category = choose_least(class_counts, ERROR_CLASSES, label_rng)
        level = choose_least(level_counts, (1, 2, 3, 4), level_rng)
        rng = random.Random(seed + 30000 + int(raw['original_index']) * 31)
        variant = rng.randrange(1000000)
        truth = {'Activity': str(raw['Activity']), 'Machine': str(raw['Resource']),
                 'WorkerID': str(raw['Worker ID']), 'Error time and Error date': str(raw['End Time'])}
        source_id = f"event-{int(raw['original_index'])}"
        if level < 3:
            source = lower_levels.lower_brief({**raw, 'ground_truth': truth,
                'true_category': category, 'report_id': source_id}, level, variant, styles)
        else:
            fault = pd.Timestamp(truth['Error time and Error date'])
            minutes, second = rng.choice(rendering.OFFSETS), rng.choice((6, 8, 12, 17))
            choices = []
            for mode in ('since_start', 'before_inspection', 'after_other_job', 'two_step'):
                delta = pd.Timedelta(minutes=minutes + (second if mode == 'two_step' else 0))
                reference = fault + delta if mode == 'before_inspection' else fault - delta
                if reference.date() == fault.date():
                    choices.append({'time_mode': mode, 'reference_time': str(reference),
                        'offset_minutes': minutes, 'second_offset_minutes': second if mode == 'two_step' else None})
            pool = frame.loc[frame.Resource.ne(truth['Machine']) & frame['Worker ID'].ne(truth['WorkerID'])]
            if pool.empty:
                raise ValueError('No event with a different machine and worker ID for the second operation.')
            other = pool.iloc[rng.randrange(len(pool))]
            context = {'Activity': str(other['Activity']), 'Machine': str(other['Resource']),
                'WorkerID': str(other['Worker ID']), 'context_type': 'synthetic_same_shift_operation'}
            tags = rng.sample(range(310, 990), 2)
            original = {'ground_truth': truth, 'context': context, 'source_report_id': source_id,
                'original_index': int(raw['original_index']), 'case_id': str(raw['Case ID']),
                'true_category': category, 'same_activity_context': truth['Activity'] == context['Activity'],
                'target_order': f'work order {tags[0]}', 'context_order': f'work order {tags[1]}'}
            source = next(r for r in upper_levels.make_pair(original, rng.choice(choices), variant) if r['difficulty_level'] == level)
            source['context_source_original_index'] = int(other['original_index'])
        source.update(report_id=f'log-l{level}-{source_id}', category_assignment='synthetic_independent_of_activity')
        sources.append(source)
    return sources


def plan(events_path, batches, seed):
    frame = read_events(events_path)
    briefs = []
    for count in batches:
        sources = make_sources(frame, briefs, count, seed)
        briefs += [v11.make_brief(source, seed) for source in sources]
    return briefs


def report_id(brief):
    return 'report-' + brief['source_report_id'].removeprefix('log-')


def options(brief, seed):
    return {'temperature': 0.9, 'top_p': 0.95, 'num_ctx': 8192, 'seed': seed + brief['original_index']}


def record(brief, text, model, digest, seed):
    return {'case_id': brief['case_id'], 'difficulty_level': brief['difficulty_level'], 'error_text': text,
            'original_index': brief['original_index'], 'report_id': report_id(brief),
            'true_category': brief['true_category'], 'ground_truth': brief['ground_truth'],
            'generation': {'generation_model': model, 'model_digest': digest,
                           'generation_options': options(brief, seed),
                           'generation_prompt': brief['generation_prompt']}}


def read_reports(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def compare(briefs, reference_path):
    reference = {r['report_id']: r['generation']['generation_prompt'] for r in read_reports(reference_path)}
    return sum(reference.get(report_id(b)) == b['generation_prompt'] for b in briefs), len(reference)


def write_prompts(briefs, path, model, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for brief in briefs:
            row = record(brief, '', model, None, seed)
            del row['error_text'], row['generation']['model_digest']
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def generate(briefs, output, host, model, seed, timeout=240):
    from ollama import Client
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {r['report_id'] for r in read_reports(output)}
    pending = [b for b in briefs if report_id(b) not in done]
    print(f'{len(done)} reports present, {len(pending)} to generate.', flush=True)
    if not pending:
        return output
    client = Client(host=host, timeout=timeout)
    installed = {m.model: m.digest for m in client.list().models}
    if model not in installed:
        raise ValueError(f'Ollama model not installed: {model}')
    digest = installed[model]
    if digest != MODEL_DIGEST:
        print(f'Note: {model} has digest {digest}; the paper dataset used {MODEL_DIGEST}.', flush=True)
    with output.open('a') as stream:
        for number, brief in enumerate(pending, len(done) + 1):
            started = time.perf_counter()
            response = client.chat(model=model, messages=[{'role': 'user', 'content': brief['generation_prompt']}],
                                   options=options(brief, seed), think=False, stream=False, keep_alive='10m')
            text = response.message.content.strip()
            if not text:
                raise RuntimeError(f'Empty answer for {report_id(brief)}; start again to retry it.')
            stream.write(json.dumps(record(brief, text, model, digest, seed), ensure_ascii=False) + '\n')
            stream.flush()
            print(f'{number}/{len(briefs)} {report_id(brief)}: {len(text.split())} words, '
                  f'{time.perf_counter() - started:.1f} s', flush=True)
    return output
