"""Check licensed input hashes and reproduce saved audience benchmark predictions."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from audience_experiment import encode, predict
from audience_features import DATA, TEXT, build_frame
from audience_models import EXPERIMENTS, LENS
from endterm import check_continuity
from prepare_audience_sources import sha256
from project import ROOT


def main():
    study = ROOT/'results/audience'
    record = json.loads((study/'training_selection.json').read_text())
    results = {row['experiment']:row for row in json.loads((study/'benchmark_metrics.json').read_text())}
    lens_path = DATA/'movielens_aggregate.csv'
    details_path, reviews_path = DATA/'tvmaze_details.jsonl', DATA/'celluloid_excerpts.jsonl'
    for name, path in [('details',details_path),('reviews',reviews_path),('lens',lens_path)]:
        if sha256(path) != record['source_hashes'][name]:
            raise ValueError(f'{name} source hash changed')
    for name, expected_digest in record['code_hashes'].items():
        if sha256(ROOT/'code'/name) != expected_digest:
            raise ValueError(f'{name} training code hash changed')
    _, _, test, _, _, continuity = check_continuity(ROOT)
    assert continuity == record['continuity']
    frame, _ = build_frame(test,details_path,reviews_path)
    frame = frame.join(pd.read_csv(lens_path).set_index('show_id',verify_integrity=True)[LENS],on='show_id')
    expected = pd.read_csv(study/'benchmark_predictions.csv')
    np.testing.assert_array_equal(frame.show_id,expected.show_id)
    np.testing.assert_array_equal(frame.high_rated,expected.high_rated)
    vectors = encode(frame,TEXT)
    checks = []
    for variant in [name for name in record['selection'] if not name.endswith('_local')]:
        selection, encoder, fitted = joblib.load(study/'models'/f'{variant}.joblib')
        assert selection == record['selection'][variant] and encoder.blocks == EXPERIMENTS[variant]
        values, predicted = predict(selection,encoder,fitted,frame,vectors)
        np.testing.assert_array_equal(predicted,expected[variant])
        np.testing.assert_allclose(values,expected[variant+'_score'],rtol=1e-7,atol=1e-8)
        assert np.isclose(f1_score(frame.high_rated,predicted),results[variant]['f1'])
        checks.append(variant)
    print(f'PASS: source hashes and {len(checks)} saved public audience models on all {len(frame)} exact test IDs.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
