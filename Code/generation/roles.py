from __future__ import annotations

import copy
import hashlib
import json
import random
import re

from . import upper_levels as paired
from . import identifiers as v8

base = v8.base
ERROR_CLASSES, MISTRAL_DIGEST = v8.ERROR_CLASSES, v8.MISTRAL_DIGEST
VERSION = 'mistral-role-balanced-v9-1'
MARKERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'


def rng_for(seed, event, purpose, identity=None):
    key = json.dumps([VERSION, seed, int(event), purpose, identity], sort_keys=True)
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))


def marker_assignments(seed, event, mechanism, tags):
    identities = sorted(tags)
    if len(identities) != 2 or len(set(identities)) != 2:
        raise ValueError('Two distinct job tags are required.')
    letters = rng_for(seed, event, 'markers-' + mechanism, identities).sample(MARKERS, 2)
    return dict(zip(identities, letters, strict=True))


def sentence_start(text):
    return text[:1].upper() + text[1:]


def replace_bindings(text, before, after):
    mapping = {}
    for old, new in zip(before, after, strict=True):
        mapping[old] = new
        mapping[old.capitalize()] = sentence_start(new)
        mapping[sentence_start(old)] = sentence_start(new)
    pattern = '|'.join(re.escape(s) for s in sorted(mapping, key=len, reverse=True))
    return re.sub(pattern, lambda m: mapping[m[0]], text)


def bindings(row, seed):
    old = [row['target_binding'], row['context_binding']]
    mechanism, event = row['binding_mechanism'], row['original_index']
    result, metadata = list(old), {}
    if mechanism in ('batch', 'inspection_sample'):
        pattern = r'batch (\d+)-[A-Z]' if mechanism == 'batch' else r'the job producing sample [A-Z]-(\d+)'
        matches = [re.fullmatch(pattern, text) for text in old]
        if not all(matches):
            raise ValueError('Unexpected V8 batch/sample bindings.')
        tags = [m[1] for m in matches]
        assigned = marker_assignments(seed, event, mechanism, tags)
        result = [f'batch {tag}-{assigned[tag]}' if mechanism == 'batch'
                  else f'the job producing sample {assigned[tag]}-{tag}' for tag in tags]
        metadata['marker_by_tag'] = assigned
    elif mechanism == 'handover':
        incoming = bool(rng_for(seed, event, 'incoming-operator').randrange(2))
        result = (['the job you took over', 'the job your colleague continued'] if incoming
                  else ['the job you continued', 'the job your colleague took over'])
        metadata['target_is_incoming_operator'] = incoming
    return old, result, metadata


def make_brief(source, seed=73):
    if source.get('text_edit') or source.get('text_variant'):
        raise ValueError('Edited report prose cannot be used as a generation source.')
    if source.get('prompt_version') == VERSION:
        raise ValueError('V9 has already been applied.')
    previous = (copy.deepcopy(source) if source.get('prompt_version') == v8.VERSION
                else v8.make_brief(source, seed))
    row = {k: copy.deepcopy(v) for k, v in previous.items() if k not in base.OUTPUT_FIELDS}
    outcome = previous['classification_evidence_plan']['outcome']
    frame = ('Use this shared starting point for the observations: '
             + v8.original.SHARED_OUTCOMES[outcome] + '. ')
    if row['symptom_instruction'].count(frame) != 1:
        raise ValueError('Expected one V8 observation framing.')
    old_instruction = row['symptom_instruction']
    row['symptom_instruction'] = old_instruction.replace(frame, '', 1)
    for field in ('generation_prompt', 'common_brief'):
        if row.get(field):
            if row[field].count(old_instruction) != 1:
                raise ValueError('Expected one symptom instruction in ' + field)
            row[field] = row[field].replace(old_instruction, row['symptom_instruction'], 1)
    row['v9_source_evidence_plan'] = copy.deepcopy(previous['classification_evidence_plan'])
    row['classification_evidence_plan'] = {'relation': previous['classification_evidence_plan']['relation'],
                                         'outcome_frame_supplied': False}

    role_plan = {}
    if row['difficulty_level'] >= 3:
        old, new, role_plan = bindings(previous, seed)
        target, context = new
        for field in ('common_brief', 'temporal_brief'):
            row[field] = replace_bindings(row[field], old, new)
        row['target_binding'], row['context_binding'] = new
        parts = row['common_brief'].split('\n\n')
        if len(parts) != 5:
            raise ValueError('Unexpected V8 upper-rung briefing sections.')
        old_style = paired.STYLES[previous['style']] + ' ' + paired.ORDERS[previous['order']] + ' '
        if not parts[1].startswith(old_style):
            raise ValueError('Unexpected V8 writing instructions.')
        situation = parts[1][len(old_style):]
        if row['binding_mechanism'] == 'handover':
            situation = ('You took over your job at shift change while your colleague continued the neighboring operation.'
                         if role_plan['target_is_incoming_operator'] else
                         'You continued your job at shift change while your colleague took over the neighboring operation.')
            situation += (' Both supplied personnel identifiers belong to the people operating after the shift change. '
                          'An outgoing operator has no supplied identifier. Reconstruct the two current assignments '
                          'and the subsequent work in your account.')
        row['style'] = rng_for(seed, row['original_index'], 'writing-style').randrange(len(paired.STYLES))
        row['order'] = rng_for(seed, row['original_index'], 'narrative-order').randrange(len(paired.ORDERS))
        parts[1] = paired.STYLES[row['style']] + ' ' + paired.ORDERS[row['order']] + ' ' + situation
        assignments = []
        for role, values, binding in [('target', row['ground_truth'], target),
                                      ('context', row['context'], context)]:
            machine = row['identifier_forms'][role + '_machine']['text']
            worker = row['identifier_forms'][role + '_worker']['text']
            equipment = f'{sentence_start(binding)} was running on {machine}, carrying out {values["Activity"]}.'
            if row['binding_mechanism'] == 'handover':
                person = (f'You are {worker}, and you are operating {binding} when the fault occurs.' if role == 'target'
                          else f'Your colleague is {worker}, and is operating {binding} at that same time.')
            else:
                person = f'The person operating {binding} was {worker}.'
            assignments.append(equipment + ' ' + person)
        target_first = bool(rng_for(seed, row['original_index'], 'assignment-order').randrange(2))
        parts[2] = ' '.join(assignments if target_first else assignments[::-1])
        ambiguous = 'The fault belongs to only the first of these jobs.'
        if parts[3].count(ambiguous) != 1:
            raise ValueError('Expected the V8 positional fault instruction.')
        parts[3] = parts[3].replace(ambiguous,
            f'The fault belongs to {target}; {context} remains routine context.', 1)
        row['common_brief'] = '\n\n'.join(parts)
        row['generation_prompt'] = '\n\n'.join((row['common_brief'], row['temporal_brief'], row['closing_instruction']))
        role_plan.update(target_binding=target, context_binding=context,
                         target_assignment_first=target_first,
                         writing_style=row['style'], narrative_order=row['order'])
        if replace_bindings(row['temporal_brief'], new, old) != previous['temporal_brief']:
            raise ValueError('V9 changed a temporal expression.')
    for field in ('ground_truth', 'context', 'identifier_forms', 'time_form', 'time_mode',
                  'reference_time', 'offset_minutes', 'second_offset_minutes', 'written_when',
                  'calendar_date_wording', 'closing_instruction', 'true_category', 'difficulty_level'):
        if row.get(field) != previous.get(field):
            raise ValueError('V9 changed a preserved field: ' + field)
    row.update(report_id='v9-' + previous['report_id'], prompt_version=VERSION,
               v9_source_report_id=previous['report_id'],
               v9_source_prompt_sha256=base.sha(previous['generation_prompt'].encode()),
               v9_role_plan=role_plan)
    if row.get('common_brief'):
        row['common_brief_sha256'] = base.sha(row['common_brief'].encode())
    return row
