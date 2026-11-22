"""Reproduce exhaustive Final searches, with resumable outer-fold results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import joblib
from joblib import Memory, parallel_backend
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from threadpoolctl import threadpool_limits

from endterm import check_continuity, scores
from final_models import teacher_grids, synopsis_grids
from project import FEATURES, positive_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/final'
MODELS = ROOT / 'models/final'
CACHE = ROOT / '.runtime/final-cache'


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(serializable(value), indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def source_fingerprint(space, continuity):
    digest = hashlib.sha256()
    for name in ['final_models.py', 'final_experiments.py', 'endterm.py', 'project.py']:
        digest.update((ROOT / 'code' / name).read_bytes())
    digest.update(json.dumps(serializable(space), sort_keys=True).encode())
    digest.update(json.dumps(continuity, sort_keys=True).encode())
    return digest.hexdigest()


def run_family(name, estimator, space, train, folds, continuity, workers):
    folder = OUT / 'searches' / name
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint(space, continuity)
    status_path = folder / 'protocol.json'
    protocol = {
        'family': name, 'grid': space, 'combinations': len(ParameterGrid(space)),
        'outer_folds': 5, 'inner_folds': 3, 'seed': 42,
        'scoring': 'positive-class F1', 'source_fingerprint': fingerprint,
        'search': 'exhaustive GridSearchCV, complete Cartesian product',
        'test_used_for_tuning': False, 'continuity': continuity,
    }
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous['source_fingerprint'] != fingerprint:
            raise RuntimeError(f'{name}: protocol or code changed. Use a new results directory.')
    else:
        save_json(status_path, protocol)
    X = train[FEATURES + ['plot_summary']]
    y = train.high_rated
    template = clone(estimator)
    if hasattr(template, 'memory'):
        template.set_params(memory=Memory(CACHE / name, verbose=0))
    records = []
    for fold, (fit_ids, val_ids) in enumerate(folds, 1):
        checkpoint = folder / f'outer_{fold}.json'
        if checkpoint.exists():
            record = json.loads(checkpoint.read_text())
            records.append(record)
            print(f'{name} outer {fold}/5: resumed F1={record["f1"]:.4f}', flush=True)
            continue
        started = time.monotonic()
        print(f'{name} outer {fold}/5: {len(ParameterGrid(space))} configurations x 3 inner folds', flush=True)
        search = GridSearchCV(template, space, scoring='f1',
                              cv=StratifiedKFold(3, shuffle=True, random_state=42),
                              n_jobs=workers, error_score='raise', return_train_score=True,
                              pre_dispatch=workers, refit=True)
        with parallel_backend('threading'), threadpool_limits(limits=1):
            search.fit(X.iloc[fit_ids], y.iloc[fit_ids])
        pd.DataFrame(search.cv_results_).to_csv(folder / f'outer_{fold}_grid.csv', index=False)
        predicted = search.predict(X.iloc[val_ids])
        ranking = positive_scores(search.best_estimator_, X.iloc[val_ids])
        record = {'model': name, 'fold': fold, 'best_params': search.best_params_,
                  'inner_best_f1': search.best_score_, 'seconds': time.monotonic()-started,
                  **scores(y.iloc[val_ids], predicted, ranking)}
        prediction = pd.DataFrame({'show_id': train.iloc[val_ids].show_id.to_numpy(),
                                   'actual': y.iloc[val_ids].to_numpy(),
                                   'predicted': predicted, 'score': ranking})
        prediction.to_csv(folder / f'outer_{fold}_predictions.csv', index=False)
        save_json(checkpoint, record)
        records.append(record)
        print(f'{name} outer {fold}/5 complete: F1={record["f1"]:.4f}, {record["seconds"]:.0f}s', flush=True)
    full_file = folder / 'full_train.json'
    if full_file.exists() and (MODELS / f'{name}.joblib').exists():
        print(f'{name}: full-training search resumed', flush=True)
    else:
        print(f'{name}: full-training exhaustive search x 5 saved folds', flush=True)
        started = time.monotonic()
        search = GridSearchCV(template, space, scoring='f1', cv=folds, n_jobs=workers,
                              error_score='raise', return_train_score=True, pre_dispatch=workers)
        with parallel_backend('threading'), threadpool_limits(limits=1):
            search.fit(X, y)
        pd.DataFrame(search.cv_results_).to_csv(folder / 'full_train_grid.csv', index=False)
        fitted = search.best_estimator_
        if hasattr(fitted, 'memory'):
            fitted.set_params(memory=None)
        joblib.dump(fitted, MODELS / f'{name}.joblib')
        selection = {'family': name, 'best_params': search.best_params_,
                     'full_train_search_f1': search.best_score_,
                     'seconds': time.monotonic()-started, 'combinations': len(ParameterGrid(space)),
                     'outer_f1_mean': float(np.mean([record['f1'] for record in records])),
                     'outer_f1_sd': float(np.std([record['f1'] for record in records], ddof=1))}
        if hasattr(fitted, 'history_'):
            pd.DataFrame(fitted.history_).to_csv(folder / 'learning_history.csv', index=False)
            selection.update(best_epoch=fitted.best_epoch_, stopped_epoch=fitted.stopped_epoch_,
                             best_validation_loss=fitted.best_validation_loss_,
                             fitting_rows=fitted.calibration_train_size_,
                             validation_rows=fitted.calibration_validation_size_)
        save_json(full_file, selection)
    pd.DataFrame(records).to_csv(folder / 'outer_scores.csv', index=False)
    print(f'{name}: finished, outer F1={np.mean([record["f1"] for record in records]):.4f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=list(teacher_grids()) + list(synopsis_grids()), required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    _, train, _, folds, _, continuity = check_continuity(ROOT)
    definitions = {**teacher_grids(), **synopsis_grids()}
    estimator, space = definitions[args.family]
    run_family(args.family, estimator, space, train, folds, continuity, args.workers)


if __name__ == '__main__':
    main()
