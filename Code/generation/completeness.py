from __future__ import annotations

import copy

from . import symptoms as original

VERSION = 'mistral-v7-level2-completeness-1'
OPENING = (
    'This account will be read on its own, without an attached shift sheet. '
    'State the full calendar date, including day, month and year, in the report. '
    'State the supplied personnel identifier as the identifier of the person operating '
    'the incident machine; it is not a job or work-order number. Keep the supplied '
    'everyday identifier and time forms, including any morning/evening or am/pm information.'
)
CLOSING = (
    'Include the incident machine, its manufacturing activity, its operator\'s personnel '
    'identifier, the full date and the supplied incident time in the prose. Make that '
    'time the start of the reported fault, not a later stop or handover. Do not introduce '
    'additional clock times for checks, earlier symptoms or shutdown. Integrate the '
    'required information naturally without a metadata list or a recap.'
)


def apply_level2_instruction(source):
    if source['difficulty_level'] != 2:
        return copy.deepcopy(source)
    if source['prompt_version'] != original.VERSION or source.get('text_edit') or source.get('text_variant'):
        raise ValueError('Expected an unedited original V7 level-2 source.')
    prompt = source['generation_prompt']
    paragraphs = prompt.split('\n\n')
    if len(paragraphs) < 3 or source['symptom_instruction'] not in prompt:
        raise ValueError('Unexpected source prompt structure.')
    rewritten = '\n\n'.join([paragraphs[0], OPENING, *paragraphs[1:-1], CLOSING, paragraphs[-1]])
    assert rewritten.replace('\n\n' + OPENING, '', 1).replace('\n\n' + CLOSING, '', 1) == prompt
    row = {k: copy.deepcopy(v) for k, v in source.items() if k not in original.base.OUTPUT_FIELDS}
    row.update(report_id='l2-complete-' + source['report_id'], source_v7_report_id=source['report_id'],
               source_v7_prompt_sha256=original.base.sha(prompt.encode()),
               prompt_version=VERSION, generation_prompt=rewritten,
               completeness_instruction={'opening': OPENING, 'closing': CLOSING})
    return row
