from __future__ import annotations

import copy
import random
from types import SimpleNamespace

from . import observations as base
from .classes import DEFINITIONS

VERSION = 'mistral-relational-symptoms-v7-1'
MISTRAL_DIGEST = '0341632adb051badc332bd814779ab931ab78619750bebe3b7c7b1401e9d70df'

MEANINGS = {
    'Air Pressure Error': 'The underlying problem is insufficient or unstable compressed-air delivery to the operation. Changes in the pneumatic action and the work must fit this cause.',
    'Lubrication Deficiency': 'The underlying problem is inadequate lubrication at moving contacts. The observations must fit impaired friction reduction, not merely an unexplained rise in process temperature.',
    'Power Outage': 'The underlying problem is an interruption of electrical supply to the affected equipment. A disruption of work must fit that loss of supply rather than merely a commanded pause.',
    'Part Misalignment': 'The underlying problem is incorrect relative alignment of the workpiece and processing geometry. Incorrect geometry is distinct from a part that loses its restraint and moves under the process load.',
    'Clamping Error': 'The underlying problem is inadequate restraint of the workpiece during the operation. The observations must fit loss of holding, not merely a securely held part in the wrong initial alignment.',
    'Control Software Glitch': 'The underlying problem is incorrect handling of commands, sequence or state by the control software. The relationship between intended operation and actual sequence must fit this cause.',
    'Cooling System Failure': 'The underlying problem is inadequate removal of process heat by the cooling system. The observations must fit impaired heat removal, not merely resistance at an inadequately lubricated moving contact.',
    'Feed Path Error': 'The underlying problem is an incorrect, interrupted or obstructed path of advancement through the operation. The observations must fit the physical progression of tool or material, not merely a generic bad finish.',
    **DEFINITIONS,
}
SHARED_OUTCOMES = (
    'a departure from expected cycle progress',
    'a change in the consistency of the operation',
    'an unexpected change in the workpiece result',
    'a recurring irregularity during ongoing work',
    'a disruption noticed through machine behaviour and its effect on the work',
)
RELATIONS = (
    'Relate the development of the observations to ordinary changes in the operating conditions.',
    'Connect what was noticed at different points in the normal work sequence.',
    'Relate an observation during the operation to where or how its effect appears in the workpiece.',
    'Use the recurrence or progression of an observation through the ongoing work to make the account informative.',
)
SYMPTOMS = (
    'Use the assigned fault as private background, not as the conclusion of the report. '
    'Write a concrete operator account whose meaning comes from relationships among '
    'observations during the work. Individual symptoms can also occur for other reasons; '
    'their combination, development and operating context must support the assigned fault. '
    'Do not merely list several textbook signs of the same fault. Do not make a diagnostic '
    'readout, explicit alarm description or named failed component the decisive clue. '
    'Do not name the category, a synonymous diagnosis or an explanation of the cause. '
    'Do not explain why other causes have been ruled out. Use ordinary observations and '
    'actions within the supplied work sequence, not a completed troubleshooting investigation. '
    'Keep informative evidence instead of replacing it with a generic malfunction. '
    'Technical equipment and material names are allowed when needed to describe what was '
    'actually observed. Choose the concrete symptoms and wording yourself; weave their '
    'relationships into the requested writing style without a fixed symptom list or diagnostic recap.'
)


def backup_templates():
    return SimpleNamespace(
        STYLES=(
            'Write a practical shift note in a tired, informal voice, mixing full sentences with short fragments.',
            'Write a developed account for maintenance explaining observations, checks and the condition left behind.',
            'Write a matter-of-fact account for the next operator, including ordinary work context and the unresolved consequence.',
            'Write a retrospective incident account explaining what led you to investigate and how you left the job.'),
        ORDERS=(
            'Start with the condition found at handover and work back through what happened.',
            'Begin with the routine work on the other job, then move to the incident.',
            'Open with the symptom on the work order, and identify its operator and equipment later in the account.',
            'Follow the work sequence, interrupting it where a check or observation matters.'))


def evidence_plan(source, seed):
    rng = random.Random(seed + 90000 + int(source['original_index']) * 17)
    return {'outcome': rng.randrange(len(SHARED_OUTCOMES)), 'relation': rng.randrange(len(RELATIONS))}


def make_brief(source, seed=73):
    if source.get('text_edit') or source.get('text_variant'):
        raise ValueError('Edited reports are not generation sources')
    if source['true_category'] not in MEANINGS or source['difficulty_level'] not in (1, 2, 3, 4):
        raise ValueError('Unknown category or extraction rung')
    original = source['generation_prompt']
    matches = [s for s in base.OLD_INSTRUCTIONS if s in original]
    if len(matches) != 1 or original.count(matches[0]) != 1:
        raise ValueError('Expected exactly one original backup symptom instruction')
    category = source['true_category']
    introductions = (f'The fault is {category}.', f'The assigned fault type is {category}.',
                     f"The fault concerned {source.get('target_binding')}: {category}.")
    if sum(original.count(s) for s in introductions) != 1:
        raise ValueError('Prompt and assigned category disagree')
    chosen = evidence_plan(source, seed)
    replacement = (
        f"Private meaning of the assigned fault: {MEANINGS[category]} "
        f"Use this shared starting point for the observations: {SHARED_OUTCOMES[chosen['outcome']]}. "
        + RELATIONS[chosen['relation']] + ' ' + SYMPTOMS)
    row = {k: copy.deepcopy(v) for k, v in source.items() if k not in base.OUTPUT_FIELDS}
    row.update(report_id='rel-v7-' + source['report_id'], paired_source_report_id=source['report_id'],
               source_prompt_version=source.get('prompt_version'), prompt_version=VERSION,
               source_prompt_sha256=base.sha(original.encode()), condition='relational_symptoms',
               classification_evidence_plan=chosen, symptom_instruction=replacement,
               original_symptom_instruction=matches[0])
    for field in ('generation_prompt', 'common_brief'):
        if source.get(field):
            if source[field].count(matches[0]) != 1:
                raise ValueError('Common prompt has a different symptom instruction')
            row[field] = source[field].replace(matches[0], replacement, 1)
            assert row[field].replace(replacement, matches[0], 1) == source[field]
    if row.get('common_brief'):
        row['common_brief_sha256'] = base.sha(row['common_brief'].encode())
    for field in ('ground_truth', 'context', 'style', 'order', 'time_form', 'time_mode',
                  'reference_time', 'offset_minutes', 'second_offset_minutes',
                  'target_binding', 'context_binding', 'temporal_brief', 'closing_instruction'):
        assert row.get(field) == source.get(field), field
    return row
