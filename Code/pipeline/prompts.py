import json
import textwrap
from pathlib import Path

REPORT_FIELD = "{error_text}"
SHOT_COUNTS = (0, 1, 3)


def prompt_key(task, shots):
    return f"{task}_{shots}shot"


def load(folder, shots=1):
    if type(shots) is not int or shots not in SHOT_COUNTS:
        raise ValueError("Shots müssen 0, 1 oder 3 sein.")
    folder = Path(folder)
    bank = json.loads((folder / "examples.json").read_text())
    templates = {}
    for task in ("extraction", "classification"):
        template = (folder / f"{task}.txt").read_text()
        for field in (REPORT_FIELD, "{examples}", "{example_note}"):
            if template.count(field) != 1:
                raise ValueError(f"Prompt {task}: exactly one {field} field is required.")
        spec = bank[task]
        if len(spec["examples"]) != 3:
            raise ValueError(f"Prompt {task}: drei feste Beispiele erforderlich.")
        examples = []
        for example in spec["examples"][:shots]:
            answer = json.dumps(
                example["answer"],
                ensure_ascii=False,
                indent=2 if task == "classification" else None,
            )
            if task == "classification":
                answer = textwrap.indent(answer, "    ")
            examples.append(
                spec["example_template"].format(report=example["report"], answer=answer)
            )
        templates[task] = template.replace("{examples}", "".join(examples)).replace(
            "{example_note}", spec["example_note"] if shots else spec["zero_note"]
        )
    return templates


def render(template, report):
    return template.replace(REPORT_FIELD, report)
