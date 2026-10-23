from __future__ import annotations

import math
import re
import time
from collections import Counter

VERSION = "matching-cache-abstention-2"
CACHE_VERSION = "ollama-embeddings-exact-text-run-only-2"


def model_key(name):
    name = str(name)
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def model_digest(models, name):
    matches = [m for m in models if model_key(m.model) == model_key(name)]
    if len(matches) != 1 or not matches[0].digest:
        raise RuntimeError(f"Kein eindeutiger Modell-Digest für {name}.")
    return matches[0].digest


def _vector(value):
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in value
        )
    ):
        raise RuntimeError("Ungültiger Embedding-Vektor; nicht gecacht.")
    return list(value)


class EmbeddingCache:

    def __init__(self, *, host, digest):
        if not digest:
            raise ValueError("A model digest is required for the run metadata.")
        self.identity = {
            "version": CACHE_VERSION,
            "host": host.rstrip("/"),
            "model_digest": digest,
            "api": "/api/embeddings",
            "options": {},
            "input": "exact UTF-8 prompt",
        }
        self.memory = {}
        self.dimension = None
        self.stats = Counter(requests=0, memory_hits=0, computed=0, failed_computations=0)

    def get(self, text, compute):
        if not isinstance(text, str):
            raise TypeError("Embedding input must be a string.")
        self.stats["requests"] += 1
        if text in self.memory:
            self.stats["memory_hits"] += 1
            return list(self.memory[text])
        try:
            vector = _vector(compute(text))
            if self.dimension is not None and len(vector) != self.dimension:
                raise RuntimeError(
                    "Embedding-Cache: Vektordimension innerhalb des Laufs widersprüchlich."
                )
        except BaseException:
            self.stats["failed_computations"] += 1
            raise
        self.dimension = len(vector)
        self.memory[text] = tuple(vector)
        self.stats["computed"] += 1
        return list(vector)

    def summary(self):
        return {"identity": self.identity, "cache_scope": "run", "counts": dict(self.stats)}

    def close(self):
        self.memory.clear()
        self.dimension = None


def parse_choice(output, n_candidates):
    if not isinstance(output, str):
        return None, "invalid_response"
    value = output.strip()
    if value.lower() == "none":
        return None, "abstained"
    if re.fullmatch(r"(?:0|[1-9]\d*)", value):
        choice = int(value)
        if 0 <= choice < n_candidates:
            return choice, "selected"
    return None, "invalid_response"


def ask_optional(core, error_text, candidates, decisions):
    if not candidates:
        return None, 0.0, 0, 0, 0
    prompt = f"""
You are an expert at matching structured factory error descriptions to event-log entries.

Task:
Decide whether the error description can be assigned to one of the listed candidates.
Choose a candidate only when the supplied information sufficiently supports that assignment.
If none is a sufficient match, or the information does not support choosing one, reply with: none
A candidate does not have to be selected. Never invent an entry outside the list.

Error:
{error_text}

Candidates:
{chr(10).join(f"[{i}] {candidate}" for i, candidate in enumerate(candidates))}

Reply with ONLY: a candidate number from 0 to {len(candidates) - 1} OR the word none.
No other text or explanation.
"""
    start = time.perf_counter()
    response = core.ollama.chat(
        model=core.LLM_PICK_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    latency = time.perf_counter() - start
    pt, ct = response.get("prompt_eval_count"), response.get("eval_count")
    if pt is None or ct is None:
        raise RuntimeError(
            "Ollama hat keine Token-Zahlen zurückgegeben (prompt_eval_count/eval_count fehlen)."
        )
    output = response["message"]["content"]
    choice, status = parse_choice(output, len(candidates))
    decisions.append({"status": status, "choice": choice, "candidate_count": len(candidates)})
    return choice, latency, int(pt), int(ct), int(pt) + int(ct)
