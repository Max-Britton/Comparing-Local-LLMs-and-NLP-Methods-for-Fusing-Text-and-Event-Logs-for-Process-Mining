import json
from . import benchmark, learning_curve, prediction_records


def validation_report(s):
    records = []
    for repetition in range(1, s.repetitions + 1):
        folder = s.output / "records"
        history = prediction_records.completed(benchmark.read_rows(folder / "validation_predictions.jsonl"))
        for condition in learning_curve.conditions(s):
            percent = condition["training_percent"]
            source = folder / "validation_subsets" / f"validation_{percent:03d}.jsonl" if percent else folder / "validation.jsonl"
            expected = {r["report_id"] for r in benchmark.read_rows(source)}
            predictions = [r for r in history if r["method"] == condition["method"]]
            if any(r["report_id"] not in expected or r.get("prediction_partition") != "validation" for r in predictions):
                raise ValueError("Falsche Partition in der Validierungsdiagnose.")
            fit = folder / (condition["fit_method"] or condition["method"]) / "training.json"
            training = json.loads(fit.read_text()) if fit.exists() else {}
            epochs = training.get("history", [])
            chosen = training.get("chosen_epoch")
            item = {
                "run": repetition, "method": condition["method"], "training_percent": percent,
                "partition": "validation", "n_expected": len(expected), "n_scored": len(predictions),
                "complete": {r["report_id"] for r in predictions} == expected and bool(expected),
                "accuracy": sum(r["correct"] for r in predictions) / len(predictions) if predictions else None,
                "chosen_epoch": chosen, "epochs_completed": len(epochs) or None,
                "chosen_epoch_at_limit": chosen == len(epochs) if epochs else None,
                "last_epoch": epochs[-1] if epochs else None,
                "selection": training.get("selection", "Fixed rules" if condition["baseline"] == "regex" else None),
                "selected_C": training.get("selected_C"),
            }
            if predictions and s.tasks == ["classification"]:
                from .selection import classification_metrics
                item.update(classification_metrics([r["true_category"] for r in predictions], [r["prediction"] for r in predictions]))
            elif predictions:
                item["mean_attribute_accuracy"] = sum(sum(r["field_correct"].values()) for r in predictions) / (4 * len(predictions))
            records.append(item)
    benchmark.atomic(s.output / "validation_summary.json", benchmark.payload({
        "purpose": "Training-budget/selection diagnostics only. These validation scores were used for model selection and are not evaluation results. A selected final epoch is a review flag, not proof that more epochs are needed.",
        "results": records,
    }))
    benchmark.save_csv(s.output / "validation_summary.csv", records)
    return records

