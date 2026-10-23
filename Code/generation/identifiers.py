from __future__ import annotations

import copy
import hashlib
import json
import random
import re

from . import symptoms as original
from . import completeness as completeness
from .rendering import _words
from .concept import worker_words
from . import classes

base = original.base
ERROR_CLASSES, MISTRAL_DIGEST = classes.ERROR_CLASSES, original.MISTRAL_DIGEST
VERSION = 'mistral-identifier-forms-v8-1'
MACHINE_NAMES = ('machine', 'unit', 'station')
MACHINE_FORMS = ('{name} {digits}', '{name} number {digits}',
                 '{name} {words}', '{name} number {words}')
WORKER_FORMS = ('operator {value}', 'worker {value}', 'operator with ID {value}',
                'operator with personnel number {value}', 'operator with badge number {value}')
DIGIT_FORMS = ('digits', 'hyphenated', 'spoken')


def draw(seed, event_index, field, identity, size):
    key = json.dumps([VERSION, seed, int(event_index), field, identity])
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16)).randrange(size)


def machine_form(identity, event_index, seed=73):
    match = re.fullmatch(r'Machine (\d+)', identity)
    if not match:
        raise ValueError(f'Unsupported machine identifier: {identity}')
    digits = match[1]
    name_index = draw(seed, event_index, 'machine-name', identity, len(MACHINE_NAMES))
    form_index = draw(seed, event_index, 'machine-format', identity, len(MACHINE_FORMS))
    text = MACHINE_FORMS[form_index].format(name=MACHINE_NAMES[name_index],
                                          digits=digits, words=_words(int(digits)))
    return {'identity': identity, 'name': name_index, 'format': form_index, 'text': text}


def worker_form(identity, event_index, seed=73):
    if not re.fullmatch(r'ID\d+', identity):
        raise ValueError(f'Unsupported worker identifier: {identity}')
    digits = identity[2:]
    wording = draw(seed, event_index, 'worker-wording', identity, len(WORKER_FORMS))
    form = draw(seed, event_index, 'worker-digits', identity, len(DIGIT_FORMS))
    values = (digits, '-'.join(digits), ' '.join(_words(int(d)) for d in digits))
    return {'identity': identity, 'wording': wording, 'digits': DIGIT_FORMS[form],
            'text': WORKER_FORMS[wording].format(value=values[form])}


def replace_assignment(row, before, after, changes):
    for field in ('generation_prompt', 'common_brief'):
        if not row.get(field):
            continue
        if row[field].count(before) != 1:
            raise ValueError(f'{row["report_id"]}: expected one identifier assignment in {field}.')
        row[field] = row[field].replace(before, after, 1)
    changes.append({'before': before, 'after': after})


def make_brief(source, seed=73):
    if source.get('text_edit') or source.get('text_variant'):
        raise ValueError('Edited report prose cannot be used as a generation source.')
    if source.get('prompt_version') == VERSION:
        raise ValueError('Identifier forms already applied.')
    if source.get('prompt_version') in (original.VERSION, completeness.VERSION):
        previous = copy.deepcopy(source)
    else:
        previous = original.make_brief(source, seed)
    if previous['difficulty_level'] == 2 and previous['prompt_version'] == original.VERSION:
        previous = completeness.apply_level2_instruction(previous)
    row = {k: copy.deepcopy(v) for k, v in previous.items() if k not in base.OUTPUT_FIELDS}
    level = row['difficulty_level']
    changes, forms = [], {}
    if level >= 2:
        truth, event = row['ground_truth'], row['original_index']
        forms['target_machine'] = machine_form(truth['Machine'], event, seed)
        forms['target_worker'] = worker_form(truth['WorkerID'], event, seed)
        if level == 2:
            number = truth['Machine'].split()[-1]
            old_machines = ('unit ' + _words(int(number)), 'station ' + number)
            old_workers = (worker_words(truth['WorkerID']), 'worker ' + '-'.join(truth['WorkerID'][2:]))
            candidates = [f'You operated {machine} as {worker};'
                          for machine in old_machines for worker in old_workers]
            found = [phrase for phrase in candidates if phrase in row['generation_prompt']]
            if len(found) != 1:
                raise ValueError('Unexpected original level-2 machine/operator assignment.')
            replace_assignment(row, found[0],
                f'You operated {forms["target_machine"]["text"]} as {forms["target_worker"]["text"]};', changes)
        else:
            context = row['context']
            forms['context_machine'] = machine_form(context['Machine'], event, seed)
            forms['context_worker'] = worker_form(context['WorkerID'], event, seed)
            for role, values, binding, old_name in (
                    ('target', truth, row['target_binding'], 'station'),
                    ('context', context, row['context_binding'], 'unit')):
                before = (f'{binding.capitalize()} was running on {old_name} '
                          f'{_words(int(values["Machine"].split()[-1]))}, carrying out ')
                after = f'{binding.capitalize()} was running on {forms[role + "_machine"]["text"]}, carrying out '
                replace_assignment(row, before, after, changes)
                old_people = ('operator ' + worker_words(values['WorkerID']),
                              'operator with personnel number ' + '-'.join(values['WorkerID'][2:]))
                found = [f'The person operating {binding} was {person}.' for person in old_people
                         if f'The person operating {binding} was {person}.' in row['generation_prompt']]
                if len(found) != 1:
                    raise ValueError('Unexpected original operator assignment.')
                replace_assignment(row, found[0],
                    f'The person operating {binding} was {forms[role + "_worker"]["text"]}.', changes)
    for field in ('generation_prompt', 'common_brief'):
        if not previous.get(field):
            continue
        restored = row[field]
        for change in reversed(changes):
            restored = restored.replace(change['after'], change['before'], 1)
        if restored != previous[field]:
            raise ValueError(f'Unexpected change outside identifier assignments: {field}')
    row.update(report_id='idforms-' + previous['report_id'], prompt_version=VERSION,
        identifier_source_report_id=previous['report_id'],
        identifier_source_prompt_sha256=base.sha(previous['generation_prompt'].encode()),
        identifier_forms=forms, identifier_replacements=changes)
    if row.get('common_brief'):
        row['common_brief_sha256'] = base.sha(row['common_brief'].encode())
    return row
