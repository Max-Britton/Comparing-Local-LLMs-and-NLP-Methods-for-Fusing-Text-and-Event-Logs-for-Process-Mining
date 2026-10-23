from __future__ import annotations

import copy
import hashlib
import json
import random

from . import roles as facts

base = facts.base
ERROR_CLASSES, MISTRAL_DIGEST = facts.ERROR_CLASSES, facts.MISTRAL_DIGEST
VERSION = 'mistral-free-narrative-v11-2'
RANDOMIZATION_VERSION = 'mistral-free-narrative-v11-1'
REGISTERS = ('Use informal workplace language.', 'Use plain, matter-of-fact language.',
             'Use a professional reporting voice.', 'Use the practical language of an experienced operator.')
FACT_FIELDS = ('original_index', 'case_id', 'category_assignment', 'difficulty_level',
    'level_name', 'ground_truth', 'context', 'context_source_original_index', 'true_category',
    'binding_mechanism', 'target_binding', 'context_binding', 'same_activity_context',
    'identifier_forms', 'written_when', 'time_form', 'time_mode', 'reference_time',
    'offset_minutes', 'second_offset_minutes', 'calendar_date_wording')


def rng_for(seed, event, purpose):
    key = json.dumps([RANDOMIZATION_VERSION, seed, int(event), purpose])
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))


def make_brief(source, seed=73):
    if source.get('text_edit') or source.get('text_variant'):
        raise ValueError('Edited reports cannot be generation sources.')
    if source.get('prompt_version') == VERSION:
        raise ValueError('V11 has already been applied.')
    previous = facts.make_brief(source, seed)
    row = {k: copy.deepcopy(previous[k]) for k in FACT_FIELDS if k in previous}
    level, gt = row['difficulty_level'], row['ground_truth']
    register = rng_for(seed, row['original_index'], 'register').choice(REGISTERS)
    assignments = {}
    if level < 3:
        machine = gt['Machine'] if level == 1 else row['identifier_forms']['target_machine']['text']
        worker = gt['WorkerID'] if level == 1 else row['identifier_forms']['target_worker']['text']
        assignments = {'machine': f'The machine you operated was {machine}.',
            'person': f'Your personnel identifier was {worker}.' if level == 1 else f'You were {worker}, operating this machine.',
            'activity': f'The manufacturing activity was {gt["Activity"]}.'}
        temporal = f'The incident time was {row["written_when"]}.'
        scope = 'Include these facts about the incident and its operator. Do not introduce additional machine or personnel identifiers or other clock times and dates.'
    else:
        for role, values in [('target', gt), ('context', row['context'])]:
            binding = row[role + '_binding']
            forms = row['identifier_forms']
            assignments[binding + '|machine'] = f'{facts.sentence_start(binding)} used {forms[role + "_machine"]["text"]}.'
            assignments[binding + '|activity'] = f'The activity for {binding} was {values["Activity"]}.'
            assignments[binding + '|person'] = f'The person operating {binding} was {forms[role + "_worker"]["text"]}.'
        temporal = previous['temporal_brief']
        for before, after in [('The fault had begun', 'The incident occurred'),
                              ('The fault began', 'The incident occurred'),
                              ('The fault on ', 'The incident on '),
                              ('routine inspection', 'inspection'), ('routine check', 'check')]:
            temporal = temporal.replace(before, after)
        scope = (f'Include the assignments of both operations. The reported incident concerns {row["target_binding"]}. '
                 'Make the connection between this incident, its work reference and the equipment and person clear. '
                 'Keep the account to these two operations and this one incident, using the supplied identifiers and times.')
        if row['binding_mechanism'] == 'handover':
            scope += ' Both personnel identifiers refer to the people operating after the handover; the colleague is operating the other job at the time of this incident.'
    order = sorted(assignments)
    rng_for(seed, row['original_index'], 'fact-order').shuffle(order)
    wording = ('Copy the activity, machine, personnel identifier and dated incident timestamp verbatim. Keep the complete YYYY/MM/DD HH:MM timestamp together exactly as supplied.'
               if level == 1 else
               'Copy the prepared identifier and time expressions verbatim, including number words and the full calendar date. Do not convert number words to digits or replace a prepared time expression with another notation. Describe the manufacturing activities precisely in your own words.')
    if level == 4:
        wording += ' Keep the incident clock time unstated: include only the reference clocks and the exact elapsed intervals linking them to the incident.'
    symptom = ('The assigned fault is private background: ' + row['true_category'] + '. '
        + facts.v8.original.MEANINGS[row['true_category']] + ' '
        'Describe concrete, compatible observations during the work and their effects, without naming the fault category or stating a diagnosis. Choose the symptoms and wording yourself.')
    parts = ['You are a machine operator writing an English error report for colleagues.',
             register, ' '.join(assignments[k] for k in order), temporal, scope, wording,
             symptom, 'Write one error report as continuous prose, without a field/value form. Choose its structure and the placement of details yourself. Return only the report.']
    row.update(report_id='v11-' + source['report_id'], source_report_id=source['report_id'],
        prompt_version=VERSION, register_instruction=register, fact_order=order,
        temporal_brief=temporal, symptom_instruction=symptom,
        generation_prompt='\n\n'.join(parts))
    return row
