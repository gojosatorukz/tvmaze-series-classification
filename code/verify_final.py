"""Check the packaged data, exhaustive searches, checkpoint and saved-model outputs."""
import os
import ast
import hashlib
from pathlib import Path
import json

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import f1_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from endterm import check_continuity
from final_models import teacher_grids, synopsis_grids
from final_text import semantic_grids
from target_features import target_grids
from project import FEATURES
from final_experiments import serializable, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main():
    _, train, test, folds, _, continuity = check_continuity(ROOT)
    definitions = {**teacher_grids(), **synopsis_grids(), **semantic_grids(), **target_grids()}
    for name, (_, grid) in definitions.items():
        folder = ROOT / 'results/final/searches' / name
        count = len(ParameterGrid(grid))
        protocol = json.loads((folder / 'protocol.json').read_text())
        if protocol['source_fingerprint'] != source_fingerprint(grid, protocol['continuity']):
            raise AssertionError(f'{name}: training source fingerprint changed.')
        if any(protocol['continuity'].get(key) != value for key, value in continuity.items()):
            raise AssertionError(f'{name}: data/split protocol differs.')
        for key, filename in [('semantic_code_sha256', 'final_text.py'),
                              ('target_encoding_code_sha256', 'target_features.py')]:
            if key in protocol['continuity'] and protocol['continuity'][key] != hashlib.sha256(
                    (ROOT / 'code' / filename).read_bytes()).hexdigest():
                raise AssertionError(f'{name}: additional feature source changed.')
        expected_grid = {json.dumps(serializable(row), sort_keys=True) for row in ParameterGrid(grid)}
        for filename in ['full_train_grid.csv'] + [f'outer_{fold}_grid.csv' for fold in range(1, 6)]:
            frame = pd.read_csv(folder / filename)
            actual_grid = {json.dumps(serializable(ast.literal_eval(value)), sort_keys=True)
                           for value in frame.params}
            if len(frame) != count or actual_grid != expected_grid:
                raise AssertionError(f'{name}: incomplete grid in {filename}')
        for fold, (_, validation_ids) in enumerate(folds, 1):
            prediction = pd.read_csv(folder / f'outer_{fold}_predictions.csv')
            validation = train.iloc[validation_ids]
            if prediction.show_id.tolist() != validation.show_id.tolist() or not np.array_equal(
                    prediction.actual, validation.high_rated):
                raise AssertionError(f'{name}: outer fold {fold} object/label mismatch.')
            record = json.loads((folder / f'outer_{fold}.json').read_text())
            if not np.isclose(f1_score(prediction.actual, prediction.predicted), record['f1'], atol=1e-12):
                raise AssertionError(f'{name}: outer fold {fold} F1 differs.')
        scores = pd.read_csv(folder / 'outer_scores.csv')
        if len(scores) != 5:
            raise AssertionError(f'{name}: expected five outer scores')
    expected = pd.read_csv(ROOT / 'results/final/test_predictions.csv')
    if expected.show_id.tolist() != test.show_id.tolist():
        raise AssertionError('Saved test object IDs differ.')
    comparison = pd.read_csv(ROOT / 'results/final/model_comparison.csv')
    selection = json.loads((ROOT / 'results/final/selection_before_test.json').read_text())
    X = test[FEATURES+['plot_summary']]
    for record in comparison.to_dict('records'):
        model_path = ROOT / 'models/final' / f'{record["family"]}.joblib'
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != selection['model_sha256'][record['family']]:
            raise AssertionError(f'{record["family"]}: frozen model file changed.')
        model = joblib.load(model_path)
        prediction = model.predict(X)
        if not np.array_equal(prediction, expected[record['family']].to_numpy()):
            raise AssertionError(f'{record["family"]}: saved-model predictions changed.')
        if not np.isclose(f1_score(test.high_rated, prediction, zero_division=0), record['test_f1'], atol=1e-12):
            raise AssertionError(f'{record["family"]}: reported F1 differs.')
    model = joblib.load(ROOT / 'models/final/mlp.joblib')
    _, val_ids = train_test_split(np.arange(len(train)), test_size=.2,
                                   stratify=train.high_rated, random_state=42)
    validation = train.iloc[val_ids]
    loss = log_loss(validation.high_rated, model.predict_proba(validation[FEATURES]),
                    sample_weight=compute_sample_weight('balanced', validation.high_rated), labels=[0, 1])
    history = pd.read_csv(ROOT / 'results/final/searches/mlp/learning_history.csv')
    if model.best_epoch_ != int(history.loc[history.validation_balanced_log_loss.idxmin(), 'epoch']):
        raise AssertionError('The selected MLP epoch is not the absolute validation minimum.')
    if not np.isclose(loss, history.validation_balanced_log_loss.min(), atol=1e-8, rtol=0):
        raise AssertionError('The loaded MLP does not retain the chosen checkpoint.')
    print(f'PASS: data and split integrity, {len(definitions)} complete searches, {len(comparison)} saved models,')
    print(f'all 468 test predictions reproduced, MLP epoch {model.best_epoch_} checkpoint verified.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
