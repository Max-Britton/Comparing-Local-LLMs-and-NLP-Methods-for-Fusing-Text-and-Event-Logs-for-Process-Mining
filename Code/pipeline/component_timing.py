import math

from .request_timing import extract


def component(wall_sec, requests=()):
    rows = list(requests)
    calls = [extract(r.get("raw_response"), r.get("latency_sec")) for r in rows]
    walls = [r["wall_time_sec"] for r in calls]
    servers = [r["ollama_total_sec"] for r in calls]
    loads = [r["ollama_load_sec"] for r in calls]
    local = wall_sec - sum(walls) if all(x is not None for x in walls) else None
    if local is not None and -1e-8 < local < 0:
        local = 0.0
    valid = local is not None and math.isfinite(local) and local >= 0
    complete = valid and all(x is not None for x in servers)
    server = sum(servers) if all(x is not None for x in servers) else None
    load = sum(loads) if all(x is not None for x in loads) else None
    without_load = (
        local + server - load if complete and load is not None and server >= load else None
    )
    return {
        "component_wall_sec": wall_sec,
        "local_python_sec": local if valid else None,
        "ollama_server_sec": server,
        "ollama_load_sec": load,
        "compute_sec": local + server if complete else None,
        "compute_without_load_sec": without_load,
        "client_transport_residual_sec": sum(walls) - server if complete else None,
        "timing_complete": complete,
        "api_calls": len(rows),
        "api_calls_with_server_time": sum(x is not None for x in servers),
        "timing_definition": "compute = local elapsed outside blocking API calls + server total_duration; residual is not pure network time",
    }


def request_timing(local_sec, raw_response, wall_sec):
    return component(
        local_sec + wall_sec, [{"raw_response": raw_response, "latency_sec": wall_sec}]
    )
