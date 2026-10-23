from __future__ import annotations

import hashlib

import pandas as pd

from .rendering import _words, render_when
from .concept import worker_words

VERSION = "mistral-paired-upper16-1"
STYLES = (
    "Write an informal shift note, with practical observations and occasional sentence fragments.",
    "Write a maintenance account explaining the checks you made and the condition you left behind.",
    "Write a matter-of-fact handover account with ordinary work context and the consequence of the stoppage.",
    "Write a retrospective incident account describing how the observations developed.",
)
ORDERS = (
    "Start with the condition left for the next shift, then explain how it arose.",
    "Begin with the other operation's progress, then connect it to the interrupted job.",
    "Open with the fault observation and establish its work context later in the account.",
    "Follow the work sequence, including the checks that mattered.",
)


def person(worker: str, worded: bool) -> str:
    return ("operator " + worker_words(worker) if worded else
            "operator with personnel number " + "-".join(worker[2:]))


def make_pair(old: dict, old_time: dict, index: int) -> list[dict]:
    gt, context = old["ground_truth"], old["context"]
    fault = pd.Timestamp(gt["Error time and Error date"])
    target_operator = person(gt["WorkerID"], bool(index % 2))
    other_operator = person(context["WorkerID"], not bool(index % 2))
    target_machine = "station " + _words(int(gt["Machine"].split()[-1]))
    other_machine = "unit " + _words(int(context["Machine"].split()[-1]))
    tag_a, tag_b = old["target_order"].split()[-1], old["context_order"].split()[-1]
    mechanism = ("work_order", "batch", "handover", "inspection_sample")[index % 4]
    if mechanism == "work_order":
        target, other = f"work order {tag_a}", f"work order {tag_b}"
        situation = "Both jobs need to be covered in the next shift's production handover. Use their work orders to connect the assignments and observations."
    elif mechanism == "batch":
        target, other = f"batch {tag_a}-K", f"batch {tag_b}-M"
        situation = "Two batches are being prepared for the next production stage. Describe the condition and progress of each batch, connecting it to the equipment and person processing it."
    elif mechanism == "handover":
        target, other = "the job you took over", "the job your colleague continued"
        situation = "You took over one job at the shift change while a colleague continued the neighboring operation. Reconstruct who handled each operation and what happened after the handover."
    else:
        target, other = f"the job producing sample Q-{tag_a}", f"the job producing sample R-{tag_b}"
        situation = "Samples from two running jobs were being checked during the shift. Connect the sample observations to their production jobs, equipment and operators in your account."
    role = ("You are a shift coordinator writing an English incident account from the operators' notes."
            if mechanism == "inspection_sample" else
            "You are the operator on the job with the fault, writing an English incident account for colleagues.")
    date = f"{fault.day} {fault:%B %Y}"
    assignments = [
        f"{target.capitalize()} was running on {target_machine}, carrying out {gt['Activity']}.",
        f"The person operating {target} was {target_operator}.",
        f"{other.capitalize()} was running on {other_machine}, carrying out {context['Activity']}.",
        f"The person operating {other} was {other_operator}.",
    ]
    if index % 2:
        assignments = assignments[2:] + assignments[:2]
    style, order = index % 4, (index * 3 + 1) % 4
    form = ("12h", "spoken", "24h")[index % 3]
    common = "\n\n".join((
        role + " This account will be read on its own, without an attached shift sheet. "
        f"All the events below took place on {date}. Include that full calendar date in the prose.",
        STYLES[style] + " " + ORDERS[order] + " " + situation,
        " ".join(assignments),
        f"The fault concerned {target}: {old['true_category']}. Describe compatible observations without using the category label. "
        f"Give {other} a concrete routine progress detail. The fault belongs to only the first of these jobs.",
        "Both people's personnel identifiers must be stated and attached to their operating roles. "
        "Spread the equipment and person assignments across the account and use the job, batch or sample references to connect them to the fault. "
        "Describe each manufacturing activity specifically enough to distinguish it from other processes. "
        "Keep the supplied machine numbers, personnel digits and full date intact; weave them into the account.",
    ))
    if old["same_activity_context"]:
        common += " Both jobs use the same manufacturing process, so their assignments distinguish which one had the fault."
    closing = "Return only the report as natural prose. Place identifying details in the account without a metadata list or a closing recap of the values."
    mode = old_time["time_mode"]
    reference = pd.Timestamp(old_time["reference_time"])
    offset = int(old_time["offset_minutes"])
    second = old_time.get("second_offset_minutes")
    second = int(second) if second is not None else None
    assert reference.date() == fault.date()
    derived = (reference - pd.Timedelta(minutes=offset) if mode == "before_inspection" else
               reference + pd.Timedelta(minutes=offset + (second or 0)))
    assert derived == fault
    reference_when, fault_when = render_when(reference, form), render_when(fault, form)
    if mode == "since_start":
        anchor = f"Work on {target} began at {reference_when}."
        direct = f"The fault began at {fault_when}."
        relative = f"The fault began exactly {_words(offset)} minutes after that work began."
    elif mode == "before_inspection":
        anchor = f"The supervisor inspected {target} at {reference_when}."
        direct = f"The fault had begun at {fault_when}."
        relative = f"The fault had begun exactly {_words(offset)} minutes before that inspection."
    elif mode == "after_other_job":
        anchor = f"The colleague completed a routine inspection of {other} at {reference_when}."
        direct = f"The fault on {target} began at {fault_when}."
        relative = f"The fault on {target} began exactly {_words(offset)} minutes after that inspection ended."
    elif mode == "two_step":
        anchor = f"Work on {target} began at {reference_when}."
        checked = reference + pd.Timedelta(minutes=offset)
        direct = f"A routine check on that job finished at {render_when(checked, form)}. The fault began at {fault_when}."
        relative = (f"A routine check on that job finished {_words(offset)} minutes after its start. "
                    f"The fault began {_words(second)} minutes after the check finished.")
    else:
        raise ValueError(mode)
    pair = []
    for level in (3, 4):
        timing = anchor + " " + (direct if level == 3 else relative)
        timing += (" Include these event times and distinguish the fault time from the other times."
                   if level == 3 else
                   " Include these reference events and exact intervals with their stated direction. Leave the incident clock time unstated.")
        prompt = common + "\n\n" + timing + "\n\n" + closing
        pair.append({
            "report_id": f"paired16-l{level}-{old['source_report_id']}",
            "source_report_id": old["source_report_id"], "original_index": old["original_index"],
            "case_id": old["case_id"], "dataset_split": "probe", "difficulty_level": level,
            "level_name": "distributed" if level == 3 else "derived", "true_category": old["true_category"],
            "ground_truth": gt, "context": context, "binding_mechanism": mechanism,
            "target_binding": target, "context_binding": other, "style": style, "order": order,
            "time_form": form, "time_mode": mode, "same_activity_context": old["same_activity_context"],
            "challenge_variant": bool(old["same_activity_context"] or (level == 4 and mode == "two_step")),
            "reference_time": str(reference), "offset_minutes": offset, "second_offset_minutes": second,
            "calendar_date_wording": date, "common_brief": common, "common_brief_sha256": hashlib.sha256(common.encode()).hexdigest(),
            "temporal_brief": timing, "closing_instruction": closing,
            "prompt_version": VERSION, "generation_prompt": prompt,
        })
    fixed = ("ground_truth", "context", "binding_mechanism", "target_binding", "context_binding",
             "style", "order", "time_form", "common_brief", "closing_instruction")
    assert all(pair[0][field] == pair[1][field] for field in fixed)
    return pair
