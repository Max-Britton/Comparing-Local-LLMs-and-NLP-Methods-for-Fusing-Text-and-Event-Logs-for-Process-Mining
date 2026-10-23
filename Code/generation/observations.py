from __future__ import annotations

import hashlib


VERSION = "mistral-observations-and-event-facts-v2"
MODEL = "mistral-medium-3.5:latest"
OLD_INSTRUCTIONS = (
    "Describe compatible observations and consequences without naming the category label.",
    "Describe compatible observations without using the category label.",
    "Convey it through compatible observations and symptoms; do not write that category label.",
)
OUTPUT_FIELDS = {
    "error_text", "raw_response", "generation_model", "model_digest", "generation_options",
    "generation_status", "generation_error", "latency_sec", "attempt", "semantic_status",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()
