"""Run nested threshold tuning and evaluate the immutable TVmaze test IDs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from audience_features import DATA, build_frame
from audience_models import AudienceEncoder, BLEND_HIST, BLEND_LR, EXPERIMENTS, LENS, PARAMETERS, fit_candidate, score_candidate
from endterm import check_continuity, scores
from final_analysis import bootstrap_delta
from final_experiments import save_json
from final_text import encode_texts
from prepare_audience_sources import sha256
from project import FEATURES, ROOT, SEED


def encode(frame, names):
    result = {}
    for column in names:
        content = frame[column].fillna('').str.strip()
        matrix = np.zeros((len(frame),384),dtype=np.float32)
        present = np.flatnonzero(content.ne(''))
        if len(present):
            matrix[present] = encode_texts(content.iloc[present].tolist())
        result[column] = matrix
    return result


def slice_vectors(vectors, positions):
    return {key:array[positions] for key,array in vectors.items()}


def threshold_for(y, values):
    precision, recall, thresholds = precision_recall_curve(y, values)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros_like(precision), where=(precision+recall)>0)
    # On an exact tie prefer the higher threshold and a smaller predicted positive set.
    peak = np.flatnonzero(np.isclose(f1[:-1], f1[:-1].max(), atol=1e-12, rtol=0))[-1]
    return float(thresholds[peak]), float(f1[peak])


def choose(experiment, train, vectors, splits):
    candidates = {name:np.full(len(train),np.nan) for name in PARAMETERS}
    for fitting, validation in splits:
        encoder = AudienceEncoder(experiment)
        fit_rows, val_rows = train.iloc[fitting], train.iloc[validation]
        fit_matrix = encoder.fit_transform(fit_rows, slice_vectors(vectors, fitting))
        val_matrix = encoder.transform(val_rows, slice_vectors(vectors, validation))
        for name, values in candidates.items():
            family, value = PARAMETERS[name]
            if family == 'blend':
                values[validation] = value*candidates[BLEND_LR][validation]+(1-value)*candidates[BLEND_HIST][validation]
            else:
                model = fit_candidate(name, fit_matrix, fit_rows)
                values[validation] = score_candidate(name, model, val_matrix)
    details = []
    for name, values in candidates.items():
        assert np.isfinite(values).all(), f'{name}: missing validation prediction'
        threshold, value = threshold_for(train.high_rated, values)
        details.append({'model':name, 'threshold':threshold, 'oof_f1':value})
    best = max(details, key=lambda row:row['oof_f1'])
    return best, details


def fit_selected(experiment, selection, frame, vectors):
    encoder = AudienceEncoder(experiment)
    matrices = encoder.fit_transform(frame, vectors)
    model = fit_candidate(selection['model'], matrices, frame)
    return encoder, model


def predict(selection, encoder, model, frame, vectors):
    values = score_candidate(selection['model'], model, encoder.transform(frame, vectors))
    return values, (values >= selection['threshold']).astype(int)


def run_variant(experiment, training, training_vectors, folds):
    outer_rows, tuning = [], []
    for index, (fitting, validation) in enumerate(folds, 1):
        left, right = training.iloc[fitting], training.iloc[validation]
        left_vectors = slice_vectors(training_vectors, fitting)
        right_vectors = slice_vectors(training_vectors, validation)
        inner = StratifiedKFold(3,shuffle=True,random_state=SEED)
        selected, candidates = choose(experiment, left, left_vectors, inner.split(left,left.high_rated))
        encoder, model = fit_selected(experiment, selected,left,left_vectors)
        values, predicted = predict(selected,encoder,model,right,right_vectors)
        value = f1_score(right.high_rated,predicted)
        outer_rows.append({'experiment':experiment,'fold':index,**selected,'outer_f1':value,
                           'outer_ap':scores(right.high_rated,predicted,values)['average_precision']})
        tuning.extend({'experiment':experiment,'split':str(index),**candidate} for candidate in candidates)
        print(f'{experiment}, outer {index}: F1 {value:.3f}, {selected["model"]}',flush=True)
    selected, candidates = choose(experiment, training,training_vectors,folds)
    tuning.extend({'experiment':experiment,'split':'full_train',**candidate} for candidate in candidates)
    fitted = fit_selected(experiment,selected,training,training_vectors)
    print(f'{experiment}, frozen setting: {selected}',flush=True)
    return selected, outer_rows, tuning, fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--details', type=Path, default=DATA/'tvmaze_details.jsonl')
    parser.add_argument('--reviews', type=Path, default=DATA/'celluloid_excerpts.jsonl')
    parser.add_argument('--lens', type=Path)
    parser.add_argument('--local-imdb', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT/'results/audience')
    parser.add_argument('--training-only', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1,7), default=3)
    parser.add_argument('--variants', nargs='+', choices=list(EXPERIMENTS))
    args = parser.parse_args()
    target, train, test, folds, _, continuity = check_continuity(ROOT)
    requested = ['catalogue','production','episode_ratings_only','first_three','first_season','all_episodes','with_reviews']
    if args.lens:
        requested.extend(['with_lens_ratings','with_rater_history'])
    if args.local_imdb:
        requested.extend(['imdb_show_only_local','imdb_only_local','cross_site_local'])
        if args.lens:
            requested.append('all_sources_local')
    experiments = args.variants or requested
    blocks = {block for name in experiments for block in EXPERIMENTS[name]}
    details_path = args.details if {'structure','episode_ratings','season_ratings'} & blocks else None
    needed_vectors = ['plot_summary'] if any(AudienceEncoder(name).use_metadata for name in experiments) else []
    needed_vectors += [column for column, block in [('episode_text','episode_text'), ('review_text','review')]
                       if block in blocks]
    frame, coverage = build_frame(target, details_path, args.reviews, args.local_imdb)
    if details_path and coverage['matched_details'] != len(target):
        raise ValueError(f'Details collection incomplete: {coverage["matched_details"]}/{len(target)}')
    if args.lens:
        frame = frame.join(pd.read_csv(args.lens).set_index('show_id',verify_integrity=True)[LENS], on='show_id')
    source_paths = {key: p for key,p in [('details',details_path),('reviews',args.reviews),
                                          ('lens',args.lens),('local_imdb',args.local_imdb)] if p}
    protocol = {'continuity':continuity,'coverage':coverage,
                'source_hashes':{key:sha256(path) for key,path in source_paths.items()},
                'code_hashes':{name:sha256(ROOT/'code'/name) for name in (
                    'audience_experiment.py','audience_features.py','audience_models.py')},
                'runtime':{'scikit_learn':sklearn.__version__,'numpy':np.__version__},
                'methods':PARAMETERS,'experiments':{name:EXPERIMENTS[name] for name in experiments},
                'nested_protocol':'Five frozen outer folds. Three inner stratified folds tune model and F1 threshold. Five saved folds tune final deployment.',
                'label':'Saved October 2026 TVmaze rating >= 8',
                'caveat':'Retrospective benchmark. Current episode and IMDb ratings may contain votes cast after the frozen target date.',
                'test_dataset':'Same 468 shows reported at earlier course stages. Variant choices and thresholds are selected from training CV.'}
    args.out.mkdir(parents=True, exist_ok=True)
    by_id = frame.set_index('show_id',drop=False,verify_integrity=True)
    training = by_id.loc[train.show_id].reset_index(drop=True)
    training_vectors = encode(training, needed_vectors)
    selections, outer_rows, tuning = {}, [], []
    estimators = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {name:executor.submit(run_variant,name,training,training_vectors,folds) for name in experiments}
        for name, result in pending.items():
            selected, cv_folds, candidate_scores, fitted = result.result()
            selections[name], estimators[name] = selected, fitted
            outer_rows.extend(cv_folds)
            tuning.extend(candidate_scores)
    outer = pd.DataFrame(outer_rows)
    cv_stats = outer.groupby('experiment').outer_f1.agg(['mean','std']).sort_values('mean',ascending=False)
    public = [name for name in cv_stats.index if not name.endswith('_local')]
    local = [name for name in cv_stats.index if name.endswith('_local')]
    selections_path = args.out/'training_selection.json'
    record = {**protocol, 'selected_utc':datetime.now(timezone.utc).isoformat(),
              'selection':selections, 'outer_cv':cv_stats.to_dict('index'),
              'public_winner':public[0] if public else None,'local_winner':local[0] if local else None}
    if selections_path.exists():
        previous = json.loads(selections_path.read_text())
        if any(previous.get(key) != json.loads(json.dumps(value)) for key,value in record.items() if key!='selected_utc'):
            raise ValueError('Selection or sources changed. Use a new output directory.')
    else:
        save_json(selections_path, record)
    outer.to_csv(args.out/'outer_cv.csv',index=False)
    pd.DataFrame(tuning).to_csv(args.out/'tuning.csv',index=False)
    print(cv_stats.to_string(),flush=True)
    if args.training_only:
        return
    heldout = by_id.loc[test.show_id].reset_index(drop=True)
    heldout_vectors = encode(heldout, needed_vectors)
    original = joblib.load(ROOT/'models/final/gradient_boosting.joblib')
    original_pred = original.predict(test[FEATURES])
    original_scores = original.predict_proba(test[FEATURES])[:,1]
    output = heldout[['show_id','high_rated']].copy()
    evaluation = [{'experiment':'final_gradient_boosting',**scores(heldout.high_rated,original_pred,original_scores)}]
    model_dir = args.out/'models'
    model_dir.mkdir(exist_ok=True)
    subgroups = []
    for experiment in experiments:
        selected = selections[experiment]
        encoder, model = estimators[experiment]
        values, predicted = predict(selected,encoder,model,heldout,heldout_vectors)
        evaluation.append({'experiment':experiment,**scores(heldout.high_rated,predicted,values),
                           'tn_fp_fn_tp':confusion_matrix(heldout.high_rated,predicted,labels=[0,1]).ravel().tolist(),
                           'paired_vs_final':bootstrap_delta(heldout.high_rated.to_numpy(),original_pred,predicted)})
        output[experiment] = predicted
        coverage_groups = {'first_three_ratings_available':heldout.first_three_rated.gt(0),
                           'episode_rating_missing':heldout.first_three_rated.eq(0),
                           'has_review':heldout.has_review.eq(1)}
        if args.lens:
            coverage_groups['has_movielens_history'] = heldout.lens_votes.notna()
        for group, present in coverage_groups.items():
            if present.sum() and heldout.high_rated[present].nunique() == 2:
                subgroups.append({'experiment':experiment,'group':group,'n':int(present.sum()),
                                  'positives':int(heldout.high_rated[present].sum()),
                                  'f1': f1_score(heldout.high_rated[present],predicted[present],zero_division=0)})
        if not experiment.endswith('_local'):
            output[experiment+'_score'] = values
        joblib.dump((selected,encoder,model),model_dir/f'{experiment}.joblib',compress=3)
        print(f'{experiment}, benchmark F1 {evaluation[-1]["f1"]:.3f}',flush=True)
    output['final_gradient_boosting'] = original_pred
    output.to_csv(args.out/'benchmark_predictions.csv',index=False)
    pd.DataFrame(subgroups).to_csv(args.out/'coverage_metrics.csv', index=False)
    save_json(args.out/'benchmark_metrics.json',evaluation)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
