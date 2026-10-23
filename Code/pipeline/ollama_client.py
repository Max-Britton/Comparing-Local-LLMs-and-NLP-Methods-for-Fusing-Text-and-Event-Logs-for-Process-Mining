import time
from contextlib import contextmanager
from pathlib import Path

from . import ollama_runtime as lifecycle
from . import request_timing as timing


@contextmanager
def benchmark_session(client, model, allowed, audit_path, host):
    with lifecycle.batch_session(client, model, audit_path, host, preserve_loaded=True):
        yield


class Transport:
    def __init__(self, client, model, *, options, audit_path):
        self.client, self.model, self.options = client, dict(model), dict(options)
        self.audit_path = Path(audit_path)
        self.before = list(client.ps().models)
        self.owned = set()
        self.active = None
        self.management_sec = 0.0
        self.events = []

    def activate(self, name):
        if self.active == name:
            return
        start = time.perf_counter()
        try:
            if self.active in self.owned:
                lifecycle.unload(self.client, self.active)
                self.events.append({"released": self.active})
                self.owned.remove(self.active)
            if lifecycle.target_was_loaded(self.client, name, self.before) is False:
                self.owned.add(name)
            self.active = name
        finally:
            self.management_sec += time.perf_counter() - start

    def embeddings(self, *, model, prompt):
        self.activate(model)
        response = self.client.embed(model=model, input=prompt, truncate=False, keep_alive="10m")
        raw = (
            response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
        )
        vectors = raw.pop("embeddings")
        if len(vectors) != 1:
            raise RuntimeError("Embedding-Antwort muss genau einen Vektor enthalten.")
        return raw | {"embedding": vectors[0], "api_endpoint": "/api/embed"}

    def chat(self, *, model, messages):
        if model != self.model["name"]:
            raise ValueError("Pass-3-Modell unterscheidet sich von der gewählten Bedingung.")
        self.activate(model)
        return self.client.chat(
            model=model,
            messages=messages,
            think=self.model["thinking"],
            options=self.options,
            stream=False,
            keep_alive="10m",
        )

    def close(self):
        try:
            for name in list(self.owned):
                lifecycle.unload(self.client, name)
                self.events.append({"released_at_end": name})
                self.owned.remove(name)
        finally:
            timing.atomic(
                self.audit_path,
                timing.as_json(
                    {
                        "policy": "pre-existing models are preserved; only models loaded by this session are released",
                        "preloaded": [m.model for m in self.before],
                        "events": self.events,
                        "model_management_sec": self.management_sec,
                    }
                ),
            )
