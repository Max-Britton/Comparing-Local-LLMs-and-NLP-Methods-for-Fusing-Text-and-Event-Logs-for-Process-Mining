from __future__ import annotations


from . import rendering as gen


VERSION = "v4-concept8-1"
STYLES = (
    "Write a concise but complete handover note for the next operator. Include what "
    "made you notice the problem and the state in which you left the work.",
    "Write a developed incident account for the next shift. Give the work context, "
    "the observations that led you to stop or investigate, and the unresolved state "
    "or consequence. Explain the connections in natural prose.",
)
ORDERS = (
    "Begin with the condition left for the next shift, then reconstruct the events.",
    "Begin with the observation that made you notice the fault; weave the work context into the account.",
    "Follow the sequence of work, noticing the fault and responding to it.",
    "Begin with the reason for documenting the incident, then explain how it developed.",
)


def worker_words(value: str) -> str:
    if not value.startswith("ID") or not value[2:].isdigit():
        raise ValueError(f"Unsupported worker ID: {value}")
    return "ID " + " ".join(gen._words(int(digit)) for digit in value[2:])
