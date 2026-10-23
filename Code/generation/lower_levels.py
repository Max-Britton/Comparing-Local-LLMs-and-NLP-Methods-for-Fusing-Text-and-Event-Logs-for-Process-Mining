from __future__ import annotations

import json

import pandas as pd

from . import rendering as gen
from .concept import worker_words

VERSION = "mistral-corpus50-extension-1"


def unpack(value):
    return json.loads(value) if isinstance(value, str) else value


def lower_brief(source, level, index, templates):
    truth = unpack(source["ground_truth"])
    style, order = (index + level - 1) % 4, (index * 3 + level) % 4
    form = ("spoken", "12h", "24h")[(index + level) % 3]
    fault = pd.Timestamp(truth["Error time and Error date"])
    when = truth["Error time and Error date"] if level == 1 else gen.render_when(fault, form)
    timing = f"The incident itself happened at {when}."
    if level == 1:
        facts = (f"You operated {truth['Machine']} as {truth['WorkerID']} performing {truth['Activity']}. "
                 f"{timing} Copy these four values exactly into the report.")
        relation = ""
    else:
        worker = worker_words(truth["WorkerID"]) if index % 2 else "worker " + "-".join(truth["WorkerID"][2:])
        machine = ("unit " + gen._words(int(truth["Machine"].split()[-1])) if index % 2 else
                   "station " + truth["Machine"].split()[-1])
        facts = f"You operated {machine} as {worker}; the manufacturing activity was {truth['Activity']}. {timing}"
        relation = ("Express the manufacturing work in your own precise words and use the supplied everyday identifier "
                    "and time forms. Preserve the identity of the activity and the identifier digits.")
    narrative = templates.ORDERS[order] if order != 1 else "Begin with the checks or observations, then explain the work context."
    prompt = "\n\n".join(x for x in (
        "You are a machine operator writing an English account for colleagues on the next shift.",
        relation, templates.STYLES[style], narrative, facts,
        f"The fault is {source['true_category']}. Describe compatible observations and consequences without naming the category label.",
        "Write a natural account using these facts, with supporting work details where useful. Use your own sentence structure. Return only the report.",
    ) if x)
    return {
        "report_id": f"ext50-l{level}-{source['report_id']}", "source_report_id": source["report_id"],
        "original_index": int(source["original_index"]), "case_id": str(source["Case ID"]),
        "dataset_split": "probe", "difficulty_level": level, "level_name": "literal" if level == 1 else "reformatted",
        "true_category": source["true_category"], "ground_truth": truth, "context": None,
        "style": style, "order": order, "time_form": "literal" if level == 1 else form,
        "time_mode": "direct", "written_when": when, "prompt_version": VERSION,
        "brief_template_version": "mistral-varied32-1", "generation_prompt": prompt,
    }
