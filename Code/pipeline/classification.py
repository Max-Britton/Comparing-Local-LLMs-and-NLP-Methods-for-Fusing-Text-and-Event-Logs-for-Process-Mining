import json
import re

ALIASES = {"Software Glitch": "Control Software Glitch"}


def parse_category(raw, labels):
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r'{\s*"Category":\s*".+?"\s*}', raw, re.DOTALL)
        try:
            parsed = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            parsed = {}
    label = parsed.get("Category") if isinstance(parsed, dict) else None
    return label if label in labels else "Unknown"
