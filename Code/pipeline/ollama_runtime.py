from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone

BENCHMARK_MODELS = {"gemma3:12b", "gemma4:31b", "mistral-medium-3.5:latest"}


def loaded_names(client):
    return [m.model for m in client.ps().models]


def target_was_loaded(client, model, before):

    def name_key(name):
        return name if ":" in name else name + ":latest"

    if any(name_key(m.model) == name_key(model) for m in before):
        return True
    if not before:
        return False
    installed = next(
        (m for m in client.list().models if name_key(m.model) == name_key(model)), None
    )
    digest = getattr(installed, "digest", None)
    other_digests = [getattr(m, "digest", None) for m in before]
    if not digest or any(not value for value in other_digests):
        return None
    return digest in other_digests


def unload(client, model, timeout=30):
    response = client.generate(model=model, keep_alive=0)
    deadline = time.monotonic() + timeout
    while model in loaded_names(client):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Ollama still lists {model} after its unload request")
        time.sleep(0.25)
    return response.model_dump(mode="json")


def save_audit(path, audit):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


@contextmanager
def batch_session(client, model, audit_path, host, *, preserve_loaded=False):
    audit = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "model": model,
        "policy": (
            "preserve pre-existing models; release only newly loaded target"
            if preserve_loaded
            else "one model per batch; unload before and in finally"
        ),
        "events": [],
        "status": "starting",
    }
    started = False
    release_target = False
    try:
        before_models = list(client.ps().models)
        before = [m.model for m in before_models]
        audit["before"] = before
        if preserve_loaded:
            audit["target_preloaded"] = target_was_loaded(client, model, before_models)
            release_target = audit["target_preloaded"] is False
        else:
            unexpected = set(before) - BENCHMARK_MODELS
            if unexpected:
                raise RuntimeError(f"Other models are loaded; left untouched: {sorted(unexpected)}")
            for name in before:
                audit["events"].append(
                    {"phase": "before", "model": name, "response": unload(client, name)}
                )
            if loaded_names(client):
                raise RuntimeError("Another model appeared before batch start; batch not started")
            release_target = True
        audit["release_target_on_exit"] = release_target
        started = True
        audit["status"] = "running"
        save_audit(audit_path, audit)
        yield audit
        audit["status"] = "finished"
    except BaseException as exc:
        audit["status"] = "failed"
        audit["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if started:
                after_batch = loaded_names(client)
                audit["after_batch"] = after_batch
                if release_target and model in after_batch:
                    audit["events"].append(
                        {"phase": "after", "model": model, "response": unload(client, model)}
                    )
                audit["after_cleanup"] = loaded_names(client)
                if not preserve_loaded and audit["after_cleanup"]:
                    raise RuntimeError(f"Models still loaded after batch: {audit['after_cleanup']}")
        except BaseException as exc:
            audit["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            audit["status"] = "cleanup_failed"
            raise
        finally:
            audit["ended_at"] = datetime.now(timezone.utc).isoformat()
            save_audit(audit_path, audit)


@contextmanager
def host_lock(host):
    yield
