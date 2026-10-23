from dataclasses import replace
import json
import shutil
import time

from . import crf, extraction, progress
from .taxonomy import ERROR_CLASSES


def validate_partitions(train, validation):
    if not train or not validation:
        raise ValueError('Model selection requires nonempty train and validation subsets.')
    for key in ('report_id', 'split_group'):
        a = {r[key] for r in train if key in r}
        b = {r[key] for r in validation if key in r}
        if a & b:
            raise ValueError(f'Train/validation overlap in {key}.')
    if len({r['report_id'] for r in train}) != len(train) or len({r['report_id'] for r in validation}) != len(validation):
        raise ValueError('Duplicate report IDs in selection data.')


def classification_metrics(truth, predictions):
    from sklearn.metrics import accuracy_score, f1_score
    return {'accuracy': float(accuracy_score(truth, predictions)),
        'macro_f1': float(f1_score(truth, predictions, labels=list(ERROR_CLASSES),
                                   average='macro', zero_division=0))}


def make_linear(name, seed, C):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline, make_union
    estimators = {
        'tfidf_logreg': LogisticRegression(C=C, max_iter=2000, class_weight='balanced', random_state=seed),
        'tfidf_svm': LinearSVC(C=C, class_weight='balanced', random_state=seed),
    }
    vectorizer = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2))
    return make_pipeline(vectorizer, estimators[name])


def fit_linear(s, output, name, train, validation, seed):
    from .benchmark import freeze, payload, digest, atomic, rows_bytes
    import joblib
    validate_partitions(train, validation)
    output.mkdir(parents=True, exist_ok=True)
    signature = {'baseline': name, 'seed': seed, 'C_grid': s.linear_c_values,
        'train_sha256': digest(rows_bytes(train)), 'validation_sha256': digest(rows_bytes(validation)),
        'selection': 'Highest validation macro-F1 over all ten classes; ties retain first C (strongest regularization). No refit on validation.'}
    freeze(output/'fit_config.json', payload(signature))
    checkpoint, finished = output/'model.joblib', output/'training.json'
    if finished.exists():
        saved = json.loads(finished.read_text())
        if digest(checkpoint.read_bytes()) != saved['checkpoint_sha256']:
            raise ValueError('Linear checkpoint changed.')
        return joblib.load(checkpoint)
    start = time.perf_counter()
    trials, best_score, best_model, chosen = [], -1., None, None
    x, y = [r['error_text'] for r in train], [r['true_category'] for r in train]
    vx, vy = [r['error_text'] for r in validation], [r['true_category'] for r in validation]
    for C in s.linear_c_values:
        begin = time.perf_counter()
        model = make_linear(name, seed, C)
        model.fit(x, y)
        scores = classification_metrics(vy, model.predict(vx))
        trials.append({'C': C, **scores, 'fit_and_selection_seconds': time.perf_counter()-begin})
        progress.message(f'{name}: C={C}, validation n={len(validation)}, macro-F1={scores["macro_f1"]:.4f}')
        if scores['macro_f1'] > best_score:
            best_score, best_model, chosen = scores['macro_f1'], model, C
        atomic(output/'selection_trials.json', payload(trials))
    joblib.dump(best_model, checkpoint)
    atomic(finished, payload({**signature, 'train_ids': [r['report_id'] for r in train],
        'validation_ids': [r['report_id'] for r in validation], 'training_n': len(train),
        'validation_n': len(validation), 'selected_C': chosen, 'validation_macro_f1': best_score,
        'fit_seconds': time.perf_counter()-start, 'checkpoint_sha256': digest(checkpoint.read_bytes()),
        'trials': trials}))
    return best_model


def extraction_metrics(pairs):
    exact, fields, n = 0, 0, 0
    for row, prediction in pairs:
        _, _, _, correct, _ = extraction.score(json.dumps(prediction), row['ground_truth'])
        exact += all(correct.values())
        fields += sum(correct.values())
        n += 1
    if not n: raise ValueError('Empty extraction validation.')
    return {'strict_accuracy': exact/n, 'mean_attribute_accuracy': fields/(n*len(extraction.FIELDS))}


def extraction_selection_key(scores):
    return scores['strict_accuracy'], scores['mean_attribute_accuracy']


def predict_crf(s, output, train, validation, pending, seed):
    from .benchmark import freeze, payload, digest, atomic, rows_bytes
    validate_partitions(train, validation)
    signature = {'grid': s.crf_reg_grid, 'seed': seed,
        'train_sha256': digest(rows_bytes(train)), 'validation_sha256': digest(rows_bytes(validation)),
        'selection': 'Raw extraction strict accuracy, then mean four-attribute accuracy on validation; exact ties retain earlier grid entry. Same fitted model for CRF and CRF+time.'}
    freeze(output/'selection_config.json', payload(signature))
    finished = output/'training.json'
    if not finished.exists():
        started = time.perf_counter()
        trials, best_key, chosen_dir = [], (-1., -1.), None
        for i, overrides in enumerate(s.crf_reg_grid):
            parameters = {**crf.PARAMETERS, **overrides}
            candidate = output/'candidates'/f'candidate_{i+1:02d}'
            settings = replace(s, tune_baselines=False, crf_parameters=parameters)
            predictions = crf.predict(settings, candidate, train, validation, validation, seed)
            scores = extraction_metrics((row, pred) for row, pred, _ in predictions)
            trials.append({'candidate': i+1, 'parameters': parameters, **scores})
            key = extraction_selection_key(scores)
            if key > best_key:best_key, chosen_dir = key, candidate
            atomic(output/'selection_trials.json', payload(trials))
            progress.message(f'CRF candidate {i+1}: validation n={len(validation)}, strict={key[0]:.4f}, attribute={key[1]:.4f}')
        chosen = json.loads((chosen_dir/'training.json').read_text())
        for name in [*chosen['checkpoint_sha256'], 'fit_config.json', 'train_annotations.json', 'validation_annotations.json']:
            shutil.copy2(chosen_dir/name, output/name)
        chosen.update({'selected_candidate_fit_seconds': chosen['fit_seconds'],
            'fit_seconds': time.perf_counter()-started, 'selection': signature['selection'],
            'selected_candidate': chosen_dir.name, 'validation_ids': [r['report_id'] for r in validation],
            'validation_n': len(validation), 'training_n': len(train), 'trials': trials})
        atomic(finished, payload(chosen))
    selected = json.loads(finished.read_text())
    yield from crf.predict(replace(s, crf_parameters=selected['parameters']), output,
                           train, validation, pending, seed)
