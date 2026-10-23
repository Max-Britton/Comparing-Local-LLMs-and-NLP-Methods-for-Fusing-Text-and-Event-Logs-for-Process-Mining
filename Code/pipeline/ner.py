from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

from . import ner_utils as ner
from . import normalization as common
from . import progress
from .training_diagnostics import isolated_inference


def annotate(rows):
    return common.annotate(rows)


def validation_metrics(model, tokenizer, rows, device, torch):
    from .selection import extraction_metrics
    records = [{"row_id": i, "text": row["error_text"]} for i, row in enumerate(rows)]
    outputs = ner.predict_entities(model, tokenizer, records, device, torch, 512)
    if len(outputs) != len(rows):
        raise ValueError("Unvollständige NER-Validierungsvorhersagen.")
    predictions = [common.normalize_prediction(r["predicted_entities"]) for r in outputs]
    return extraction_metrics(zip(rows, predictions))


def training_metrics(model, tokenizer, rows, device, torch):
    with isolated_inference(model, torch, device):
        scores = validation_metrics(model, tokenizer, rows, device, torch)
    return {'train_accuracy': scores['strict_accuracy'], 'train_n': len(rows)}


def predict(s, output, train, validation, pending, seed):
    from .benchmark import atomic, digest, freeze, payload
    from .selection import validate_partitions, extraction_selection_key
    validate_partitions(train, validation)

    torch, transformers = ner.require_transformer_dependencies()
    ner.set_global_seed(seed, torch)
    torch.set_num_threads(4)
    device = ner.choose_device(torch, s.ner_device)
    checkpoint = output / "checkpoint"
    finished = output / "training.json"
    freeze(
        output / "fit_config.json",
        payload(
            {
                "surface_version": common.VERSION,
                "train_sha256": digest(payload(train)),
                "validation_sha256": digest(payload(validation)),
                "base_model": Path(s.ner_model).name,
                "seed": seed,
                "epochs": s.ner_epochs,
                "batch_size": s.ner_batch_size,
                "learning_rate": s.ner_lr,
                "training_diagnostics": "Epoch-end strict four-attribute accuracy on all training reports before time rules; eval mode, preserved RNG; diagnostic only.",
                "selection": "Strict validation extraction accuracy, then mean attribute accuracy; exact ties retain the earlier epoch. Token loss is diagnostic only.",
            }
        ),
    )
    model = tokenizer = None
    try:
        if finished.exists():
            saved = json.loads(finished.read_text())
            for name, sha in saved["checkpoint_sha256"].items():
                if digest((checkpoint / name).read_bytes()) != sha:
                    raise ValueError("Gespeicherter NER-Checkpoint wurde verändert.")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                checkpoint, local_files_only=True, trust_remote_code=False
            )
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                checkpoint, local_files_only=True, trust_remote_code=False
            )
        else:
            setup_started = time.perf_counter()
            tokenizer = transformers.AutoTokenizer.from_pretrained(s.ner_model, use_fast=True, local_files_only=True, trust_remote_code=False)
            if not tokenizer.is_fast:
                raise ValueError("NER benötigt einen schnellen Tokenizer mit Zeichenpositionen.")
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                s.ner_model,
                num_labels=len(ner.BIO_LABELS),
                id2label=ner.ID2LABEL,
                label2id=ner.LABEL2ID,
                ignore_mismatched_sizes=True, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            )
            model.to(device)
            setup_seconds = time.perf_counter() - setup_started
            started = time.perf_counter()
            annotated = {"train": annotate(train), "validation": annotate(validation)}
            for name, records in annotated.items():
                freeze(output / (name + "_annotations.json"), payload(records))
            base = ner.make_dataset_class(torch)

            class Dataset(base):
                def __getitem__(self, index):
                    record = self.records[index]
                    item = self.tokenizer(
                        record["text"],
                        truncation=True,
                        max_length=self.max_length,
                        return_offsets_mapping=True,
                    )
                    offsets = item.pop("offset_mapping")
                    end = max(b for a, b in offsets)
                    visible = [span for span in record["spans"] if span["end"] <= end]
                    item["labels"] = common.token_labels(offsets, visible)
                    cut = [span for span in record["spans"] if span["start"] < end < span["end"]]
                    for i, (a, b) in enumerate(offsets):
                        if any(a < span["end"] and b > span["start"] for span in cut):
                            item["labels"][i] = -100
                    item["row_id"] = record["row_id"]
                    visible_types = {span["label"] for span in visible}
                    item["unresolved"] = [
                        ner.LABEL2ID[prefix + "-" + label]
                        for label, status in record["annotation_status"].items()
                        if status != "resolved" or label not in visible_types
                        for prefix in ("B", "I")
                    ]
                    return item

            base_collate = ner.make_collator(tokenizer, torch)

            def collate(items):
                unresolved = [item.pop("unresolved") for item in items]
                return base_collate(items) | {"unresolved": unresolved}

            usable = {
                name: [r for r in records if r["spans"]] for name, records in annotated.items()
            }
            if not usable["train"]:
                raise ValueError(
                    "Keine auflösbaren NER-Trainingsspannen. Annotationen prüfen oder NER deaktivieren."
                )
            loaders = {
                name: torch.utils.data.DataLoader(
                    Dataset(records, tokenizer, 512),
                    batch_size=s.ner_batch_size,
                    shuffle=name == "train",
                    collate_fn=collate,
                )
                for name, records in usable.items()
                if records
            }
            optimizer = torch.optim.AdamW(model.parameters(), lr=s.ner_lr, weight_decay=0.01)

            def loss_for(batch, with_count=False):
                unresolved = batch.pop("unresolved")
                batch.pop("row_ids")
                labels = batch.pop("labels").to(device)
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                mask = torch.zeros_like(logits, dtype=torch.bool)
                for index, indices in enumerate(unresolved):
                    mask[index, :, indices] = True
                logits = logits.masked_fill(mask, -1e4)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
                )
                count = int((labels != -100).sum().item())
                return (loss, count) if with_count else loss

            history, best_state, best_key, chosen_epoch = [], None, (-1., -1.), None
            for epoch in progress.iterate(
                range(1, s.ner_epochs + 1),
                s,
                desc=f"NER Training ({len(train)} Texte)",
                unit="Epoche",
            ):
                model.train()
                losses = []
                for batch in progress.iterate(
                    loaders["train"],
                    s,
                    desc=f"Epoche {epoch}/{s.ner_epochs}",
                    unit="Batch",
                    leave=False,
                ):
                    optimizer.zero_grad()
                    loss = loss_for(batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError("NER-Trainingsverlust ist nicht endlich.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    losses.append(loss.item())
                val_loss = None
                if "validation" in loaders:
                    model.eval()
                    with torch.no_grad():
                        values = [
                            (lambda pair: (pair[0].item(), pair[1]))(loss_for(batch, with_count=True))
                            for batch in progress.iterate(
                                loaders["validation"],
                                s,
                                desc="NER Validierung",
                                unit="Batch",
                                leave=False,
                            )
                        ]
                    val_loss = sum(loss*n for loss,n in values) / sum(n for _,n in values)
                    if not math.isfinite(val_loss):
                        raise RuntimeError("NER-Validierungsverlust ist nicht endlich.")
                scores = validation_metrics(model, tokenizer, validation, device, torch)
                key = extraction_selection_key(scores)
                if key > best_key:
                    best_key, chosen_epoch = key, epoch
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                diagnostic_started = time.perf_counter()
                train_scores = training_metrics(model, tokenizer, train, device, torch)
                diagnostic_seconds = time.perf_counter() - diagnostic_started
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": sum(losses) / len(losses),
                        "validation_loss": val_loss,
                        "validation_n": len(validation),
                        **scores,
                        **train_scores,
                        "training_diagnostics_seconds": diagnostic_seconds,
                    }
                )
                atomic(output / "training_history.json", payload(history))
                progress.message(f"NER Epoche {epoch}/{s.ner_epochs}: {history[-1]}")
            if best_state is not None:
                model.load_state_dict(best_state)
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            atomic(
                finished,
                payload(
                    {
                        "base_model": Path(s.ner_model).name,
                        "seed": seed,
                        "base_commit": getattr(model.config, "_commit_hash", None),
                        "training_seconds": time.perf_counter() - started,
                        "training_diagnostics_seconds": sum(h["training_diagnostics_seconds"] for h in history),
                        "training_diagnostics": "Epoch-end strict accuracy on all training reports before time rules; no selection on training scores.",
                        "history": history,
                        "model_setup_seconds": setup_seconds,
                        "setup_policy": "Model/tokenizer loading and possible HF downloads timed separately; training excludes setup.",
                        "chosen_epoch": chosen_epoch or s.ner_epochs,
                        "selection": "Highest strict validation extraction accuracy; tie: highest mean attribute accuracy; full tie: earlier epoch. Token-weighted validation loss is diagnostic only.",
                        "requested_counts": {k: len(v) for k, v in annotated.items()},
                        "usable_counts": {k: len(v) for k, v in usable.items()},
                        "train_ids": [r["report_id"] for r in train],
                        "validation_ids": [r["report_id"] for r in validation],
                        "training_n": len(train), "validation_n": len(validation),
                        "device": str(device),
                        "annotation_coverage": {
                            k: common.coverage(v) for k, v in annotated.items()
                        },
                        "checkpoint_sha256": {
                            p.name: digest(p.read_bytes())
                            for p in checkpoint.iterdir()
                            if p.is_file()
                        },
                        "annotation_policy": "Weak ground-truth spans on train/validation only; unresolved labels masked; no eval annotation.",
                        "normalization": common.VERSION
                        + "; no assumed year; fixed public process catalog.",
                    }
                ),
            )
        model.to(device)
        for index, row in enumerate(pending):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            inference_start = time.perf_counter()
            record = {"row_id": index, "text": row["error_text"]}
            output_row = ner.predict_entities(model, tokenizer, [record], device, torch, 512)[0]
            prediction = common.normalize_prediction(output_row["predicted_entities"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            yield row, prediction, time.perf_counter() - inference_start
    finally:
        model = tokenizer = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
