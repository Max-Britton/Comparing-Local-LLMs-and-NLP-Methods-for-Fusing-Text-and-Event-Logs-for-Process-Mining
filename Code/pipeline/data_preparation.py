from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "InputData" / "Raw" / "Production_Data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "InputData" / "Prepared" / "Prepared_Production_Data.csv"
DEFAULT_CANDIDATE_OUTPUT = PROJECT_ROOT / "InputData" / "Prepared" / "Error_Event_Candidates.csv"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "InputData" / "Prepared" / "Error_Event_Filter_Report.json"

ALLOWED_ERROR_ACTIVITIES = (
    "Turning & Milling",
    "Round Grinding",
    "Laser Marking",
    "Lapping",
    "Wire Cut",
    "Flat Grinding",
)

ERROR_EVENT_KEY = ("Process Step", "Machine", "End Time", "Worker ID")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_activity(value: object) -> str:
    activity = normalize_text(value)
    return re.sub(r"^SETUP\s+", "", activity, flags=re.IGNORECASE)


def process_step(value: object) -> str:
    activity = normalize_activity(value)
    activity = re.sub(r"\s*-\s*Machine\s*\d+[A-Za-z]*\s*$", "", activity, flags=re.IGNORECASE)
    return re.sub(r"\s*-\s*Machine\s*$", "", activity, flags=re.IGNORECASE).strip()


def extract_machine_id(value: object) -> pd.NA | str:
    match = re.search(r"\bMachine\s*(\d+[A-Za-z]?)\b", normalize_text(value), re.I)
    return match.group(1).upper() if match else pd.NA


def prepare_event_log(data: pd.DataFrame) -> pd.DataFrame:
    required = {"Case ID", "Activity", "Resource", "Start Timestamp", "Complete Timestamp"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {', '.join(missing)}")

    prepared = data.copy()
    if "original_index" in prepared.columns:
        raise ValueError("Source already contains reserved column 'original_index'.")

    prepared.insert(0, "original_index", range(len(prepared)))
    prepared.insert(
        prepared.columns.get_loc("Activity") + 1,
        "Activity Raw",
        prepared["Activity"].copy(),
    )
    prepared.insert(
        prepared.columns.get_loc("Resource") + 1,
        "Resource Raw",
        prepared["Resource"].copy(),
    )
    prepared["Start Time Parsed"] = pd.to_datetime(prepared["Start Timestamp"], errors="coerce")
    prepared["End Time Parsed"] = pd.to_datetime(prepared["Complete Timestamp"], errors="coerce")
    prepared["End Time"] = prepared["End Time Parsed"].dt.strftime("%Y/%m/%d %H:%M")
    prepared["Duration Minutes"] = (
        (prepared["End Time Parsed"] - prepared["Start Time Parsed"]).dt.total_seconds().div(60)
    )

    prepared["Activity Normalized"] = prepared["Activity Raw"].map(normalize_activity)
    prepared["Process Step"] = prepared["Activity Raw"].map(process_step)
    prepared["Resource Normalized"] = prepared["Resource Raw"].map(normalize_text)
    activity_machine = prepared["Activity Raw"].map(extract_machine_id)
    resource_machine = prepared["Resource Raw"].map(extract_machine_id)
    prepared["Machine ID"] = resource_machine.combine_first(activity_machine)
    prepared["Machine"] = prepared["Machine ID"].map(
        lambda value: f"Machine {value}" if pd.notna(value) else pd.NA
    )

    activity = prepared["Activity Normalized"]
    resource = prepared["Resource Normalized"]
    prepared["Is Machine Event"] = prepared["Machine ID"].notna()
    prepared["Is Quality Event"] = activity.str.contains(
        r"\bQ\.?C\.?\b|quality|inspection", case=False, regex=True
    ) | resource.str.contains(r"quality\s*check", case=False, regex=True)
    prepared["Is Rework Event"] = activity.str.contains(
        r"\brework\b|\bfix\b", case=False, regex=True
    )

    qty_rejected = pd.to_numeric(prepared.get("Qty Rejected", 0), errors="coerce").fillna(0)
    qty_mrb = pd.to_numeric(prepared.get("Qty for MRB", 0), errors="coerce").fillna(0)
    rework_flag = prepared.get("Rework", pd.Series(index=prepared.index, dtype=object))
    prepared["Has Quality Signal"] = (
        qty_rejected.gt(0)
        | qty_mrb.gt(0)
        | rework_flag.astype(str).str.strip().str.upper().eq("Y")
        | prepared["Is Rework Event"]
    )

    source_columns = [
        "Activity Raw"
        if column == "Activity"
        else "Resource Raw"
        if column == "Resource"
        else column
        for column in data.columns
    ]
    prepared["Is Exact Source Duplicate"] = prepared.duplicated(subset=source_columns, keep=False)
    prepared["Exact Duplicate Group"] = (
        prepared.groupby(source_columns, dropna=False, sort=False).ngroup()
        if source_columns
        else pd.Series(range(len(prepared)), index=prepared.index)
    )

    prepared["Activity"] = prepared["Process Step"]
    prepared["Resource"] = prepared["Machine"]
    return prepared


def annotate_error_candidates(
    prepared: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(set(ERROR_EVENT_KEY).difference(prepared.columns))
    if missing:
        raise ValueError(
            "Prepared event log is missing candidate-key columns: " + ", ".join(missing)
        )

    annotated = prepared.copy()
    complete_key = pd.Series(True, index=annotated.index)
    for column in ERROR_EVENT_KEY:
        complete_key &= annotated[column].notna()
        complete_key &= annotated[column].map(normalize_text).ne("")

    annotated["Is Allowed Error Activity"] = annotated["Process Step"].isin(
        ALLOWED_ERROR_ACTIVITIES
    )
    annotated["Has Complete Error Key"] = complete_key
    annotated["Is Potential Error Event"] = (
        annotated["Is Allowed Error Activity"]
        & annotated["Is Machine Event"].fillna(False)
        & annotated["Has Complete Error Key"]
    )

    annotated["Error Match Key Group Size"] = pd.Series(pd.NA, index=annotated.index, dtype="Int64")
    annotated["Is Ambiguous Error Match"] = False
    annotated["Is Error Candidate Eligible"] = False

    potential = annotated.loc[annotated["Is Potential Error Event"]].copy()
    if potential.empty:
        return annotated, potential

    grouped = potential.groupby(list(ERROR_EVENT_KEY), dropna=False, sort=True)
    group_sizes = grouped["original_index"].transform("size").astype("Int64")
    annotated.loc[potential.index, "Error Match Key Group Size"] = group_sizes
    ambiguous_indices = potential.index[group_sizes.gt(1)]
    eligible_indices = potential.index[group_sizes.eq(1)]
    annotated.loc[ambiguous_indices, "Is Ambiguous Error Match"] = True
    annotated.loc[eligible_indices, "Is Error Candidate Eligible"] = True

    candidates = annotated.loc[eligible_indices].copy()
    candidates = candidates.sort_values(
        ["Process Step", "End Time Parsed", "Machine", "Worker ID", "original_index"]
    ).reset_index(drop=True)

    if candidates.duplicated(subset=list(ERROR_EVENT_KEY)).any():
        raise AssertionError("Candidate pool contains an ambiguous error-event key.")
    if not candidates["Error Match Key Group Size"].eq(1).all():
        raise AssertionError("Candidate pool contains a non-unique matching key.")
    if not candidates["Process Step"].isin(ALLOWED_ERROR_ACTIVITIES).all():
        raise AssertionError("Candidate pool contains a disallowed activity.")
    return annotated, candidates


def build_filter_report(
    source: pd.DataFrame, prepared: pd.DataFrame, candidates: pd.DataFrame
) -> dict[str, Any]:
    potential = prepared.loc[prepared["Is Potential Error Event"]]
    eligible = prepared.loc[prepared["Is Error Candidate Eligible"]]
    ambiguous = prepared.loc[prepared["Is Ambiguous Error Match"]]
    ambiguous_groups = ambiguous.drop_duplicates(subset=list(ERROR_EVENT_KEY))
    per_activity: dict[str, dict[str, int]] = {}
    for activity in ALLOWED_ERROR_ACTIVITIES:
        per_activity[activity] = {
            "potential_source_rows": int(potential["Process Step"].eq(activity).sum()),
            "matchable_candidate_events": int(eligible["Process Step"].eq(activity).sum()),
            "excluded_ambiguous_rows": int(ambiguous["Process Step"].eq(activity).sum()),
            "ambiguous_key_groups": int(ambiguous_groups["Process Step"].eq(activity).sum()),
        }

    return {
        "source_rows": int(len(source)),
        "prepared_rows": int(len(prepared)),
        "all_source_rows_preserved": bool(len(source) == len(prepared)),
        "allowed_error_activities": list(ALLOWED_ERROR_ACTIVITIES),
        "error_event_key": list(ERROR_EVENT_KEY),
        "potential_error_rows": int(len(potential)),
        "matchable_candidate_events": int(len(candidates)),
        "excluded_ambiguous_rows": int(len(ambiguous)),
        "ambiguous_key_groups": int(len(ambiguous_groups)),
        "largest_ambiguous_group": (
            int(ambiguous["Error Match Key Group Size"].max()) if len(ambiguous) else 0
        ),
        "per_activity": per_activity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_CANDIDATE_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    prepared = prepare_event_log(source)
    prepared, candidates = annotate_error_candidates(prepared)
    report = build_filter_report(source, prepared, candidates)

    for path in (args.output, args.candidate_output, args.report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    candidates.to_csv(args.candidate_output, index=False)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved full prepared event log: {args.output}")
    print(f"Saved uniquely matchable candidate pool: {args.candidate_output}")
    print(f"Saved filter report: {args.report_output}")
    print(f"Rows preserved: {len(prepared)}/{len(source)}")
    print(f"Potential error rows: {report['potential_error_rows']}")
    print(f"Uniquely matchable candidate events: {len(candidates)}")
    print(f"Ambiguous rows excluded: {report['excluded_ambiguous_rows']}")


if __name__ == "__main__":
    main()
