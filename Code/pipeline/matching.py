from __future__ import annotations

import json
import math
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dateutil.parser import parse as parse_datetime
from tqdm import tqdm

from . import request_timing as timing
from .matching_prompt import VERSION as PROMPT_VERSION
from .matching_prompt import ask_with_report
from .matching_support import ask_optional

VERSION = "three-pass-matching-1"
FIELDS = ("Activity", "Machine", "Error time and Error date", "WorkerID")
EVENT_FIELDS = ("Activity", "Resource", "End Time", "Worker ID")
SCORE_FIELDS = ("Activity", "Machine", "Date", "Time", "WorkerID")


@dataclass(frozen=True)
class Settings:
    cosine_threshold: float = 0.99
    minimum_margin: float = 0.0
    tie_tolerance: float = 1e-9
    time_margin_seconds: float = 60.0
    top_k: int = 10
    minimum_direct_attributes: int = 2
    pass1_normalize: bool = False
    score_mode: str = "hybrid"
    hybrid_threshold: float = 0.75
    activity_weight: float = 1.0
    machine_weight: float = 1.0
    date_weight: float = 5.0
    time_weight: float = 1.0
    worker_weight: float = 1.0
    time_scale_seconds: float = 300.0

    def weights(self):
        return dict(
            zip(
                SCORE_FIELDS,
                (
                    self.activity_weight,
                    self.machine_weight,
                    self.date_weight,
                    self.time_weight,
                    self.worker_weight,
                ),
            )
        )

    def threshold(self):
        return {"whole_embedding": self.cosine_threshold, "hybrid": self.hybrid_threshold}[
            self.score_mode
        ]

    def validate(self):
        if (
            not -1 <= self.cosine_threshold <= 1
            or not math.isfinite(self.minimum_margin)
            or self.minimum_margin < 0
            or not math.isfinite(self.tie_tolerance)
            or self.tie_tolerance < 0
            or not math.isfinite(self.time_margin_seconds)
            or self.time_margin_seconds < 0
            or type(self.top_k) is not int
            or self.top_k < 2
            or type(self.minimum_direct_attributes) is not int
            or not 1 <= self.minimum_direct_attributes <= 4
            or type(self.pass1_normalize) is not bool
        ):
            raise ValueError("Ungültige Matching-Einstellungen.")
        if (
            self.score_mode not in {"whole_embedding", "hybrid"}
            or not 0 < self.hybrid_threshold <= 1
            or any(
                type(w) not in (int, float) or not math.isfinite(w) or w < 0
                for w in self.weights().values()
            )
            or sum(self.weights().values()) <= 0
            or not math.isfinite(self.time_scale_seconds)
            or self.time_scale_seconds <= 0
        ):
            raise ValueError("Ungültiger Attributvergleich, Gewichte oder Zeitskala.")


def clean(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return "" if text.casefold() in {"not specified", "unknown", "none", "null", "n/a"} else text


def normalized_text(value):
    return " ".join(clean(value).casefold().split())


def identifier(value, kind):
    text = normalized_text(value)
    prefix = (
        r"(?:machine|unit|station)" if kind == "machine" else r"(?:worker\s*id|operator\s*id|id)"
    )
    match = re.fullmatch(rf"(?:{prefix}\s*[:#-]?\s*)?([0-9](?:[\s_-]*[0-9])*)", text)
    if not match:
        return text
    digits = re.sub(r"\D", "", match[1])
    return str(int(digits)) if kind == "machine" else digits


def timestamp(value):
    text = clean(value)
    full_date = (
        re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}(?=[T\s]|$)", text)
        or re.search(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b", text)
        or re.search(
            r"\b(?:\d{1,2}\s+[A-Za-z]+\s+\d{4}|"
            r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}|"
            r"\d{4}\s+[A-Za-z]+\s+\d{1,2})\b",
            text,
        )
    )
    if not full_date or not re.search(r"(?<!\d)\d{1,2}:\d{2}", text):
        return None
    try:
        value = parse_datetime(text, fuzzy=False, default=datetime(2000, 1, 1))
        return value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


@dataclass(frozen=True)
class TemporalParts:
    date: str | None
    clock_seconds: float | None
    date_present: bool
    time_present: bool


def temporal_parts(value):
    text = clean(value)
    if not text:
        return TemporalParts(None, None, False, False)
    dates = list(
        re.finditer(
            r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|"
            r"\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}|"
            r"\d{4}\s+[A-Za-z]+\s+\d{1,2})(?!\d)",
            text,
        )
    )
    clocks = list(
        re.finditer(
            r"(?<!\d)\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s*[ap]m)?"
            r"(?:\s*(?:Z|[+-]\d{2}(?::?\d{2})?))?(?!\d)",
            text,
            flags=re.IGNORECASE,
        )
    )
    if len(dates) > 1 or len(clocks) > 1 or not (dates or clocks):
        return TemporalParts(None, None, True, True)

    def parse_token(matches):
        if not matches:
            return None
        try:
            return parse_datetime(matches[0][0], fuzzy=False, default=datetime(2000, 1, 1))
        except (ValueError, TypeError, OverflowError):
            return None

    date, clock = parse_token(dates), parse_token(clocks)
    calendar = date.date().isoformat() if date is not None else None
    seconds = None
    if clock is not None:
        try:
            combined = datetime.combine(
                date.date() if date is not None else datetime(2000, 1, 1).date(), clock.timetz()
            )
            utc = combined.replace(tzinfo=combined.tzinfo or timezone.utc).astimezone(timezone.utc)
            seconds = utc.hour * 3600 + utc.minute * 60 + utc.second + utc.microsecond / 1e6
            if calendar is not None:
                calendar = utc.date().isoformat()
        except (ValueError, OverflowError):
            seconds = None
            if clock.tzinfo is not None:
                calendar = None
    return TemporalParts(calendar, seconds, bool(dates), bool(clocks))


def parse_extraction(raw, *, normalize_values=True):
    text = "" if raw is None else str(raw).strip()
    if not clean(text):
        return {}, "empty_extraction"
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return {}, "invalid_json"
    if not isinstance(value, dict) or any(
        value.get(key) is not None
        and (
            type(value[key]) not in (str, int, float)
            or isinstance(value[key], float)
            and not math.isfinite(value[key])
        )
        for key in FIELDS
    ):
        return {}, "invalid_schema"
    info = {
        key: (
            clean(value.get(key))
            if normalize_values
            else str(value[key])
            if clean(value.get(key))
            else ""
        )
        for key in FIELDS
    }
    return info, "ok" if any(info.values()) else "no_attributes"


def embedding_text(info):
    return " | ".join(f"{label}: {info.get(key, '')}" for key, label in zip(FIELDS, EVENT_FIELDS))


class RecordedOllama:

    def __init__(self, client, output):
        self.client = client
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.report_id = None

    def _call(self, task, **kwargs):
        start = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        raw, error = {}, ""
        try:
            response = getattr(self.client, task)(**kwargs)
            raw = (
                response.model_dump(mode="json")
                if hasattr(response, "model_dump")
                else dict(response)
            )
            return response
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, Exception):
                raise RuntimeError(
                    f"{task}-Aufruf für {kwargs['model']} fehlgeschlagen: {error}"
                ) from exc
            raise
        finally:
            wall = time.perf_counter() - start
            raw.pop("embedding", None)
            record = {
                "method": "llm_matching",
                "task": task,
                "report_id": self.report_id,
                "model": kwargs["model"],
                "request": kwargs,
                "raw_response": raw,
                "latency_sec": wall,
                "api_error": error,
                "request_started_at": started_at,
                "request_finished_at": datetime.now(timezone.utc).isoformat(),
            }
            record.update(timing.extract(raw, wall))
            self.records.append(record)
            with (self.output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def embeddings(self, **kwargs):
        return self._call("embeddings", **kwargs)

    def chat(self, **kwargs):
        return self._call("chat", **kwargs)

    def save(self):
        timing.save_csv(
            self.output / "requests.csv",
            [
                {k: v for k, v in r.items() if k not in {"request", "raw_response"}}
                for r in self.records
            ],
        )
        timing.atomic(
            self.output / "summary.json",
            timing.as_json(
                {
                    task: timing.summarize([r for r in self.records if r["task"] == task])
                    for task in ("embeddings", "chat")
                }
            ),
        )


class Catalog:
    def __init__(self, events, settings):
        self.events = events
        self.settings = settings
        self.ids = np.arange(len(events), dtype=int)
        self.raw = {
            field: np.array([str(e[column]) for e in events], dtype=object)
            for field, column in zip(FIELDS, EVENT_FIELDS)
        }
        times = [timestamp(e["End Time"]) for e in events]
        self.normalized = {
            "Activity": np.array([normalized_text(e["Activity"]) for e in events], dtype=object),
            "Machine": np.array(
                [identifier(e["Resource"], "machine") for e in events], dtype=object
            ),
            "WorkerID": np.array(
                [identifier(e["Worker ID"], "worker") for e in events], dtype=object
            ),
            "Error time and Error date": np.array(
                [v if v is not None else np.nan for v in times], dtype=float
            ),
        }
        parts = [temporal_parts(e["End Time"]) for e in events]
        self.dates = np.array([p.date for p in parts], dtype=object)
        self.clocks = np.array(
            [p.clock_seconds if p.clock_seconds is not None else np.nan for p in parts], dtype=float
        )
        self.texts = [
            embedding_text(dict(zip(FIELDS, (e[c] for c in EVENT_FIELDS)))) for e in events
        ]

    def select(self, info, *, normalize=True):
        provided = [key for key in FIELDS if info.get(key)]
        counts = np.zeros(len(self.events), dtype=int)
        for key in provided:
            if not normalize:
                mask = self.raw[key] == info[key]
            elif key == "Error time and Error date":
                value = timestamp(info[key])
                mask = (
                    np.abs(self.normalized[key] - value) <= self.settings.time_margin_seconds
                    if value is not None
                    else np.zeros(len(self.events), dtype=bool)
                )
            else:
                value = (
                    identifier(info[key], "machine")
                    if key == "Machine"
                    else identifier(info[key], "worker")
                    if key == "WorkerID"
                    else normalized_text(info[key])
                )
                mask = self.normalized[key] == value
            counts += mask
        best = int(counts.max()) if len(counts) else 0
        ids = self.ids[counts == best] if provided else np.array([], dtype=int)
        return {
            "ids": ids,
            "provided": len(provided),
            "agreement": best,
            "strict": bool(provided) and best == len(provided),
            "mode": (
                "strict"
                if provided and best == len(provided)
                else "max_agreement"
                if best
                else "no_attribute_agreement"
            ),
        }


class Ranker:
    def __init__(self, catalog, cache, client, model):
        self.catalog, self.cache, self.client, self.model = catalog, cache, client, model
        self.matrix = None
        self.ready = np.zeros(len(catalog.events), dtype=bool)
        self.normalized = {}
        self.attribute_matrices = {}
        self.attribute_ready = {}
        self.query_temporal = {}
        self.last_components = None

    def vector(self, text):
        vector = self.cache.get(
            text, lambda t: self.client.embeddings(model=self.model, prompt=t).get("embedding")
        )
        if text not in self.normalized:
            arr = np.asarray(vector, dtype=float)
            norm = np.linalg.norm(arr)
            if not np.isfinite(norm) or norm == 0:
                raise RuntimeError("Embedding hat keine gültige Kosinus-Norm.")
            self.normalized[text] = arr / norm
        return self.normalized[text]

    def top(self, info, ids):
        self.last_components = None
        if not len(ids):
            return []
        if self.catalog.settings.score_mode != "whole_embedding":
            scores, components = self.attribute_scores(info, ids)
            order = np.argsort(-scores, kind="stable")[: self.catalog.settings.top_k]
            self.last_components = {
                int(ids[pos]): {key: float(values[pos]) for key, values in components.items()}
                for pos in order
            }
            return [(int(ids[pos]), float(scores[pos])) for pos in order]
        query = self.vector(embedding_text(info))
        if self.matrix is None:
            self.matrix = np.empty((len(self.catalog.events), len(query)), dtype=float)
        for idx in ids[~self.ready[ids]]:
            self.matrix[idx] = self.vector(self.catalog.texts[idx])
            self.ready[idx] = True
        scores = self.matrix[ids] @ query
        order = np.argsort(-scores, kind="stable")[: self.catalog.settings.top_k]
        return [(int(ids[pos]), float(scores[pos])) for pos in order]

    def attribute_cosines(self, field, value, ids):
        column = EVENT_FIELDS[FIELDS.index(field)]
        query_text = f"{column}: {value}"
        query = self.vector(query_text)
        if field not in self.attribute_matrices:
            self.attribute_matrices[field] = np.zeros(
                (len(self.catalog.events), len(query)), dtype=float
            )
            self.attribute_ready[field] = np.zeros(len(self.catalog.events), dtype=bool)
        matrix, ready = self.attribute_matrices[field], self.attribute_ready[field]
        for idx in ids[~ready[ids]]:
            candidate = clean(self.catalog.events[idx][column])
            if candidate:
                matrix[idx] = self.vector(f"{column}: {candidate}")
            ready[idx] = True
        return np.clip(matrix[ids] @ query, 0.0, 1.0)

    def attribute_scores(self, info, ids):
        settings = self.catalog.settings
        weights = settings.weights()
        components = {}
        temporal_value = info.get("Error time and Error date", "")
        if temporal_value not in self.query_temporal:
            self.query_temporal[temporal_value] = temporal_parts(temporal_value)
        parts = self.query_temporal[temporal_value]
        for key in SCORE_FIELDS:
            present = (
                parts.date_present
                if key == "Date"
                else parts.time_present
                if key == "Time"
                else bool(info.get(key))
            )
            if not present or weights[key] == 0:
                continue
            if key == "Activity":
                scores = self.attribute_cosines(key, info[key], ids)
            elif key == "Date":
                scores = (
                    (self.catalog.dates[ids] == parts.date).astype(float)
                    if parts.date is not None
                    else np.zeros(len(ids), dtype=float)
                )
            elif key == "Time":
                if parts.clock_seconds is None:
                    scores = np.zeros(len(ids), dtype=float)
                else:
                    delta = np.abs(self.catalog.clocks[ids] - parts.clock_seconds)
                    scores = np.nan_to_num(
                        1.0 / (1.0 + delta / settings.time_scale_seconds), nan=0.0
                    )
            else:
                value = identifier(info[key], "machine" if key == "Machine" else "worker")
                scores = (self.catalog.normalized[key][ids] == value).astype(float)
            components[key] = scores
        weight_sum = sum(weights[key] for key in components)
        combined = sum(
            (weights[key] * values for key, values in components.items()),
            np.zeros(len(ids), dtype=float),
        )
        return combined / weight_sum if weight_sum else combined, components


def direct_choice(top, selection, settings):
    if not top:
        return None
    margin = top[0][1] - top[1][1] if len(top) > 1 else math.inf
    if (
        top[0][1] >= settings.threshold()
        and margin > settings.tie_tolerance
        and margin >= settings.minimum_margin
    ):
        return top[0][0]
    return None


def run_matching(
    reports,
    events,
    client,
    *,
    embedding_model,
    matching_model,
    cache,
    settings=Settings(),
    order_seed=None,
    progress=True,
    use_original_report=True,
):
    if type(use_original_report) is not bool:
        raise ValueError("MIT_ORIGINALBERICHT muss True oder False sein.")
    if use_original_report:
        for report in reports:
            if not isinstance(report.get("error_text"), str) or not report["error_text"].strip():
                raise ValueError("Matching mit Originalbericht benötigt für jeden Bericht error_text.")
    start = time.perf_counter()
    settings.validate()
    catalog = Catalog(events, settings)
    ranker = Ranker(catalog, cache, client, embedding_model)
    order = list(range(len(reports)))
    if order_seed is not None:
        random.Random(order_seed).shuffle(order)
    results, plans, infos = [], {}, {}
    used = set()
    decisions = []
    core = SimpleNamespace(ollama=client, LLM_PICK_MODEL=matching_model)

    def event_ids(ids):
        return [str(events[int(idx)]["original_index"]) for idx in ids]

    def assign(i, idx, method):
        results[i]["matched_event_id"] = str(events[idx]["original_index"])
        results[i]["match_method"] = method
        used.add(idx)

    def available(i):
        return np.array([idx for idx in plans[i]["ids"] if int(idx) not in used], dtype=int)

    def active():
        return [i for i in order if infos[i][1] == "ok" and results[i]["matched_event_id"] is None]

    for i, report in enumerate(reports):
        info, status = parse_extraction(report["raw_output"])
        infos[i] = info, status
        results.append(
            {
                "report_id": report["report_id"],
                "matched_event_id": None,
                "match_method": status if status != "ok" else "no_match",
                "parse_status": status,
                "match_latency_sec": 0.0,
                "match_prompt_tokens": 0,
                "match_completion_tokens": 0,
                "match_total_tokens": 0,
                "llm_status": None,
                "filter_candidate_ids": [],
                "top10_candidate_ids": None,
                "top10_scores": None,
                "llm_candidate_ids": None,
                "score_mode": settings.score_mode,
                "top10_attribute_scores": None,
                "llm_attribute_scores": None,
            }
        )

    for i in tqdm(active(), desc="Pass 1/3: Attributfilter", disable=not progress):
        entry = time.perf_counter()
        info = infos[i][0]
        pass1_info = (
            info
            if settings.pass1_normalize
            else parse_extraction(reports[i]["raw_output"], normalize_values=False)[0]
        )
        first_selection = catalog.select(pass1_info, normalize=settings.pass1_normalize)
        selection = first_selection if settings.pass1_normalize else catalog.select(info)
        plans[i] = selection
        result = results[i]
        result.update(
            filter_candidate_ids=event_ids(selection["ids"]),
            filter_mode=selection["mode"],
            agreeing_attributes=selection["agreement"],
            provided_attributes=selection["provided"],
            pass1_normalized=settings.pass1_normalize,
            pass1_candidate_ids=event_ids(first_selection["ids"]),
            pass1_filter_mode=first_selection["mode"],
        )
        if (
            first_selection["strict"]
            and first_selection["provided"] >= settings.minimum_direct_attributes
            and len(first_selection["ids"]) == 1
            and int(first_selection["ids"][0]) not in used
        ):
            assign(i, int(first_selection["ids"][0]), "single_candidate")
        result["match_latency_sec"] += time.perf_counter() - entry

    proposals = []
    for i in tqdm(active(), desc="Pass 2/3: Ähnlichkeitsvergleich", disable=not progress):
        entry = time.perf_counter()
        client.report_id = reports[i]["report_id"]
        ids = available(i)
        top = ranker.top(infos[i][0], ids)
        results[i].update(
            top10_candidate_ids=event_ids([idx for idx, _ in top]),
            top10_scores=[score for _, score in top],
            top10_attribute_scores=(
                [ranker.last_components[idx] for idx, _ in top]
                if ranker.last_components is not None
                else None
            ),
        )
        candidate = direct_choice(top, plans[i], settings)
        if candidate is not None:
            proposals.append((top[0][1], i, candidate))
        results[i]["match_latency_sec"] += time.perf_counter() - entry
    proposals.sort(key=lambda p: p[0], reverse=True)
    for _, i, idx in proposals:
        if idx not in used:
            assign(
                i,
                idx,
                "cosine_direct" if settings.score_mode == "whole_embedding" else "attribute_direct",
            )

    for i in tqdm(active(), desc="Pass 3/3: LLM-Auswahl", disable=not progress):
        entry = time.perf_counter()
        client.report_id = reports[i]["report_id"]
        top = ranker.top(infos[i][0], available(i))
        result = results[i]
        result["llm_candidate_ids"] = event_ids([idx for idx, _ in top])
        result["llm_attribute_scores"] = (
            [ranker.last_components[idx] for idx, _ in top]
            if ranker.last_components is not None
            else None
        )
        if not top:
            result["match_method"] = "no_unused_candidates"
        else:
            info = infos[i][0]
            structured = "\n".join(f"{key}: {info[key]}" for key in FIELDS)
            if use_original_report:
                choice, latency, pt, ct, tt = ask_with_report(
                    core,
                    reports[i]["error_text"],
                    structured,
                    [catalog.texts[idx] for idx, _ in top],
                    decisions,
                )
            else:
                choice, latency, pt, ct, tt = ask_optional(
                    core, structured, [catalog.texts[idx] for idx, _ in top], decisions
                )
            result.update(
                match_prompt_tokens=pt,
                match_completion_tokens=ct,
                match_total_tokens=tt,
                llm_latency_sec=latency,
                llm_status=decisions[-1]["status"],
            )
            if choice is None:
                result["match_method"] = decisions[-1]["status"]
            else:
                assign(i, top[choice][0], "llm_choice")
        result["match_latency_sec"] += time.perf_counter() - entry

    return results, {
        "use_original_report": use_original_report,
        "prompt_version": PROMPT_VERSION if use_original_report else "attributes-only-1",
        "component_total_sec": time.perf_counter() - start,
        "cache": cache.summary(),
        "order_seed": order_seed,
        "score_mode": settings.score_mode,
        "pass1_normalized": settings.pass1_normalize,
        "attribute_weights": settings.weights(),
        "score_threshold": settings.threshold(),
        "time_scale_seconds": settings.time_scale_seconds,
        "date_comparison": "exact UTC calendar date (year-month-day)",
        "time_comparison": "absolute seconds-of-day difference; calendar date scored separately",
        "report_order": [reports[i]["report_id"] for i in order],
    }
