import gc
import json
import math
import time
from pathlib import Path

from . import ner_utils, progress
from .selection import classification_metrics, validate_partitions
from .taxonomy import ERROR_CLASSES
from .training_diagnostics import isolated_inference


def synchronize(torch, device):
    if device.type == 'mps': torch.mps.synchronize()
    elif device.type == 'cuda': torch.cuda.synchronize()


def training_metrics(model, features, collate, batch_size, labels, device, torch):
    predicted, actual = [], []
    with isolated_inference(model, torch, device):
        for start in range(0, len(features), batch_size):
            batch = collate([dict(row) for row in features[start:start + batch_size]])
            actual.extend(batch.pop('labels').tolist())
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())
    scores = classification_metrics([labels[i] for i in actual], [labels[i] for i in predicted])
    return {'train_accuracy': scores['accuracy'], 'train_macro_f1': scores['macro_f1'],
            'train_n': len(actual)}


def predict(s, output, train, validation, pending, seed):
    from .benchmark import atomic, digest, freeze, payload, rows_bytes
    validate_partitions(train, validation)
    torch, transformers = ner_utils.require_transformer_dependencies()
    ner_utils.set_global_seed(seed, torch)
    torch.set_num_threads(4)
    device = ner_utils.choose_device(torch, s.bert_device)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, finished = output/'checkpoint', output/'training.json'
    labels = list(ERROR_CLASSES)
    label2id = {name: i for i, name in enumerate(labels)}
    signature = {'base_model': Path(s.bert_model).name, 'seed': seed, 'epochs': s.bert_epochs,
        'batch_size': s.bert_batch_size, 'learning_rate': s.bert_lr, 'weight_decay': 0.01,
        'max_length': 512, 'train_sha256': digest(rows_bytes(train)),
        'validation_sha256': digest(rows_bytes(validation)), 'labels': labels,
        'training_diagnostics': 'Epoch-end accuracy and macro-F1 on all training reports; eval mode, preserved RNG; diagnostic only.',
        'selection': 'Highest validation macro-F1; tie: lowest example-weighted validation loss; remaining tie: earlier epoch. Training only; no refit on validation.'}
    freeze(output/'fit_config.json', payload(signature))
    model = tokenizer = None
    try:
        setup_started = time.perf_counter()
        if finished.exists():
            saved = json.loads(finished.read_text())
            for name, sha in saved['checkpoint_sha256'].items():
                if digest((checkpoint/name).read_bytes()) != sha:
                    raise ValueError('BERT checkpoint changed.')
            tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False, use_safetensors=True)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(s.bert_model, local_files_only=True, trust_remote_code=False)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                s.bert_model, num_labels=len(labels), id2label=dict(enumerate(labels)),
                label2id=label2id, local_files_only=True, trust_remote_code=False,
                use_safetensors=True)
            model.to(device)
            setup_seconds = time.perf_counter()-setup_started
            def encode(rows):
                encoded = tokenizer([r['error_text'] for r in rows], truncation=True, max_length=512)
                return [{**{key: value[i] for key, value in encoded.items()},
                         'labels': label2id[row['true_category']]} for i, row in enumerate(rows)]
            collate = transformers.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
            train_loader = torch.utils.data.DataLoader(encode(train), batch_size=s.bert_batch_size,
                shuffle=True, generator=torch.Generator().manual_seed(seed), collate_fn=collate)
            val_loader = torch.utils.data.DataLoader(encode(validation), batch_size=s.bert_batch_size,
                shuffle=False, collate_fn=collate)
            optimizer = torch.optim.AdamW(model.parameters(), lr=s.bert_lr, weight_decay=0.01)
            total_steps = len(train_loader)*s.bert_epochs
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
            history, best_key, best_state, chosen_epoch = [], (-1., -math.inf), None, None
            started = time.perf_counter()
            for epoch in range(1, s.bert_epochs+1):
                model.train()
                train_loss_sum, train_n = 0., 0
                for batch in progress.iterate(train_loader, s, desc=f'BERT epoch {epoch}/{s.bert_epochs}', unit='batch', leave=False):
                    batch = {key: value.to(device) for key, value in batch.items()}
                    optimizer.zero_grad()
                    loss = model(**batch).loss
                    if not torch.isfinite(loss): raise RuntimeError('Non-finite BERT training loss.')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step(); scheduler.step()
                    n = len(batch['labels']); train_n += n; train_loss_sum += loss.item()*n
                model.eval()
                val_loss_sum, val_n, pred_ids, true_ids = 0., 0, [], []
                with torch.inference_mode():
                    for batch in val_loader:
                        batch = {key: value.to(device) for key, value in batch.items()}
                        outputs = model(**batch)
                        n = len(batch['labels']); val_n += n; val_loss_sum += outputs.loss.item()*n
                        pred_ids.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
                        true_ids.extend(batch['labels'].cpu().tolist())
                val_loss = val_loss_sum/val_n
                if not math.isfinite(val_loss): raise RuntimeError('Non-finite BERT validation loss.')
                scores = classification_metrics([labels[i] for i in true_ids], [labels[i] for i in pred_ids])
                key = (scores['macro_f1'], -val_loss)
                if key > best_key:
                    best_key, chosen_epoch = key, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                diagnostic_started = time.perf_counter()
                train_scores = training_metrics(model, train_loader.dataset, collate,
                                                s.bert_batch_size, labels, device, torch)
                diagnostic_seconds = time.perf_counter() - diagnostic_started
                history.append({'epoch': epoch, 'train_loss': train_loss_sum/train_n,
                                'validation_loss': val_loss, 'validation_n': val_n, **scores,
                                **train_scores, 'training_diagnostics_seconds': diagnostic_seconds})
                atomic(output/'training_history.json', payload(history))
                progress.message(f'BERT epoch {epoch}: train accuracy={train_scores["train_accuracy"]:.4f}, validation accuracy={scores["accuracy"]:.4f}, macro-F1={scores["macro_f1"]:.4f}, loss={val_loss:.4f}; selected epoch={chosen_epoch}')
            model.load_state_dict(best_state)
            best_state = None
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            atomic(finished, payload({**signature, 'chosen_epoch': chosen_epoch, 'device': str(device),
                'training_seconds': time.perf_counter()-started, 'model_setup_seconds': setup_seconds,
                'training_diagnostics_seconds': sum(h['training_diagnostics_seconds'] for h in history),
                'train_ids': [r['report_id'] for r in train],
                'validation_ids': [r['report_id'] for r in validation],
                'training_n': len(train), 'validation_n': len(validation), 'history': history,
                'checkpoint_sha256': {p.name: digest(p.read_bytes()) for p in checkpoint.iterdir() if p.is_file()}}))
            optimizer = scheduler = train_loader = val_loader = None
        model.to(device); model.eval()
        for row in pending:
            synchronize(torch, device)
            started = time.perf_counter()
            inputs = tokenizer(row['error_text'], truncation=True, max_length=512, return_tensors='pt').to(device)
            with torch.inference_mode():
                predicted = model(**inputs).logits.argmax(dim=-1).item()
            synchronize(torch, device)
            yield row, labels[predicted], time.perf_counter()-started
    finally:
        model = tokenizer = None
        gc.collect()
        if device.type == 'mps': torch.mps.empty_cache()
        elif device.type == 'cuda': torch.cuda.empty_cache()
