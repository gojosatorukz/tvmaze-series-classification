"""Compare larger training sets and simple IMDb-based models with a fresh test."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from audience_experiment import fit_selected, predict as predict_audience, threshold_for
from endterm import check_continuity, scores
from final_experiments import save_json
from prepare_audience_sources import imdb_lookup, sha256
from project import FEATURES, ROOT, load_raw

DATA = ROOT/'data/audience_scaleup'
OUT = ROOT/'results/audience_scaleup'
CORE = ['imdb_show_rating','show_log_votes']
EPISODE = ['ep_mean','ep_median','ep_std','ep_high_share','ep_weighted_mean','ep_q10','ep_q90',
           'ep_log_count','ep_log_votes','first_three_mean','first_three_log_votes','first_three_high_share',
           'pilot','episode_minus_show','cross_vote_ratio']
METADATA = ['premiere_year','runtime_minutes','genre_count','is_streaming']
BLOCKS = {'show': CORE, 'show_and_episodes': CORE+EPISODE,
          'show_episodes_and_metadata': CORE+EPISODE+METADATA}


def model_for(name):
    imputer = SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    if name == 'logistic':
        return make_pipeline(imputer,StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1000))
    if name == 'smooth_logistic':
        return make_pipeline(imputer,SplineTransformer(n_knots=5,knots='uniform',degree=2,extrapolation='constant'),
                             LogisticRegression(C=.25,class_weight='balanced',max_iter=1000))
    if name == 'trees':
        return make_pipeline(imputer,ExtraTreesClassifier(n_estimators=170,min_samples_leaf=12,
                                                            class_weight='balanced',random_state=42,n_jobs=1))
    options = dict(max_iter=100,min_samples_leaf=25,l2_regularization=3.,early_stopping=False,random_state=42)
    if name == 'regression':
        return HistGradientBoostingRegressor(max_leaf_nodes=12,**options)
    return HistGradientBoostingClassifier(max_leaf_nodes=7 if name=='boost_shallow' else 20,
                                         class_weight='balanced',**options)


def cohort(name):
    path = DATA/f'{name}.csv'
    manifest_path = DATA/f'{name}_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if sha256(path) != manifest['cohort_sha256']:
        raise ValueError(f'{name}: published source hash changed')
    for page in manifest['pages']:
        raw_path = DATA/'index'/f"page_{page['page']:03d}.json"
        if sha256(raw_path) != page['sha256']:
            raise ValueError(f'{raw_path.name}: published source hash changed')
    return pd.read_csv(path,keep_default_na=False,na_values={'premiere_year':[''],'runtime_minutes':['']})


def imdb_data(ids, files):
    ratings = pd.read_csv(files/'imdb-title-ratings.tsv.gz',sep='\t')
    episode_index = pd.read_csv(files/'imdb-title-episode.tsv.gz',sep='\t',na_values='\\N')
    merged = episode_index.merge(ids,left_on='parentTconst',right_on='imdb')
    merged = merged[merged.seasonNumber.gt(0) & merged.episodeNumber.gt(0)]
    position = merged.groupby(['show_id','seasonNumber','episodeNumber']).tconst.transform('size')
    labelled = merged[position.eq(1)].merge(ratings,on='tconst')
    labelled['high'] = labelled.averageRating.ge(8)
    labelled['weight'] = np.log1p(labelled.numVotes)
    labelled['weighted_rating'] = labelled.averageRating*labelled.weight
    group = labelled.groupby('show_id')
    episode_features = group.agg(ep_mean=('averageRating','mean'),ep_median=('averageRating','median'),
                                  ep_std=('averageRating','std'),ep_high_share=('high','mean'))
    episode_features['ep_weighted_mean'] = group.weighted_rating.sum()/group.weight.sum()
    episode_features['ep_q10'] = group.averageRating.quantile(.10)
    episode_features['ep_q90'] = group.averageRating.quantile(.90)
    episode_features['ep_log_count'] = np.log1p(group.size())
    episode_features['ep_log_votes'] = np.log1p(group.numVotes.sum())
    early = labelled[labelled.seasonNumber.eq(1) & labelled.episodeNumber.le(3)]
    early_group = early.groupby('show_id')
    episode_features['first_three_mean'] = early_group.averageRating.mean()
    episode_features['first_three_log_votes'] = np.log1p(early_group.numVotes.sum())
    episode_features['first_three_high_share'] = early_group.high.mean()
    episode_features['pilot'] = early[early.episodeNumber.eq(1)].set_index('show_id').averageRating
    show = ids.merge(ratings,left_on='imdb',right_on='tconst',validate='one_to_one').set_index('show_id')
    result = show[['averageRating']].rename(columns={'averageRating':'imdb_show_rating'})
    result['show_log_votes'] = np.log1p(show.numVotes)
    result = result.join(episode_features,validate='one_to_one')
    result['episode_minus_show'] = result.ep_weighted_mean-result.imdb_show_rating
    result['cross_vote_ratio'] = result.ep_log_votes-result.show_log_votes
    return result


def add_features(frame, data):
    rows = frame.join(data,on='show_id').reset_index(drop=True)
    rows['genre_count'] = rows.genres.map(lambda value:len(json.loads(value)))
    rows['is_streaming'] = rows.distribution.isin(['Streaming','Hybrid']).astype(float)
    return rows


def fit(features, method, rows):
    model = model_for(method)
    model.fit(rows[BLOCKS[features]],rows.rating if method=='regression' else rows.high_rated)
    return model


def predict(model, features, method, rows):
    matrix = rows[BLOCKS[features]]
    return model.predict(matrix) if method=='regression' else model.predict_proba(matrix)[:,1]


def evaluate(rows,values,threshold):
    predicted = (values>=threshold).astype(int)
    counts = confusion_matrix(rows.high_rated,predicted,labels=[0,1]).ravel().tolist()
    return {**scores(rows.high_rated,predicted,values),'tn_fp_fn_tp':counts}


def add_evaluation(tested, regime, dataset, rows, values, threshold):
    tested.append({'regime':regime,'dataset':dataset,**evaluate(rows,values,threshold)})
    # This diagnostic searches with known evaluation labels, after freezing the actual model.
    _, maximum = threshold_for(rows.high_rated,values)
    tested.append({'regime':regime,'dataset':dataset+'_ranking_diagnostic',
                   'max_f1_over_thresholds_given_labels':maximum})


def select_candidates(old,extra,folds):
    methods = ['logistic','smooth_logistic','trees','boost_shallow','boost_deep','regression']
    extra_folds = list(StratifiedKFold(5,shuffle=True,random_state=194).split(extra,extra.high_rated))
    settings = []
    for regime in ('original','expanded'):
        for block in BLOCKS:
            for method in methods:
                old_scores, extra_scores = np.empty(len(old)),np.empty(len(extra))
                for (old_fit,old_val),(new_fit,new_val) in zip(folds,extra_folds):
                    fitting = old.iloc[old_fit]
                    if regime=='expanded':
                        fitting = pd.concat([fitting,extra.iloc[new_fit]],ignore_index=True)
                    estimator = fit(block,method,fitting)
                    old_scores[old_val] = predict(estimator,block,method,old.iloc[old_val])
                    extra_scores[new_val] = predict(estimator,block,method,extra.iloc[new_val])
                optimized_threshold, _ = threshold_for(pd.concat([old.high_rated,extra.high_rated]),
                                                       np.r_[old_scores,extra_scores])
                old_result = f1_score(old.high_rated,old_scores>=optimized_threshold,zero_division=0)
                extra_result = f1_score(extra.high_rated,extra_scores>=optimized_threshold,zero_division=0)
                settings.append({'regime':regime,'features':block,'method':method,
                                 'threshold':optimized_threshold,'oof_f1_old':old_result,'oof_f1_extra':extra_result,
                                 'oof_f1_mean':(old_result+extra_result)/2})
            print(f'Tuned {regime} / {block}',flush=True)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('imdb_files',type=Path)
    args = parser.parse_args()
    _, training, benchmark, folds, _, continuity = check_continuity(ROOT)
    additional = cohort('training_extra')
    sealed = cohort('sealed')
    assert not set(training.show_id) & set(additional.show_id)
    assert not set(pd.concat([training,benchmark,additional]).show_id) & set(sealed.show_id)
    raw, _ = load_raw()
    for path in sorted((DATA/'index').glob('page_*.json')):
        raw.extend(json.loads(path.read_text()))
    all_rows = pd.concat([training,benchmark,additional,sealed],ignore_index=True)
    ids = imdb_lookup(all_rows,raw)
    data = imdb_data(ids,args.imdb_files)
    train_a, train_b = add_features(training,data),add_features(additional,data)
    settings = select_candidates(train_a,train_b,folds)
    settings.sort(key=lambda row:row['oof_f1_mean'],reverse=True)
    best = [next(row for row in settings if row['regime']==regime) for regime in ('original','expanded')]
    primary = max(best,key=lambda row:row['oof_f1_mean'])
    size_control = next(row for row in settings if row['regime']=='expanded'
                        and row['features']==primary['features'] and row['method']==primary['method'])
    OUT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'frozen_selection.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'primary':primary,
              'selected_per_regime':best,'same_model_expanded':size_control,
              'old_train':len(training),'extra_train':len(additional),
              'continuity':continuity,'sealed_count':len(sealed),
              'input_hashes':{name:sha256(args.imdb_files/name) for name in ('imdb-title-ratings.tsv.gz','imdb-title-episode.tsv.gz')},
              'selection':'Model and threshold from five out-of-fold predictions in the two training catalogues. Candidate choice maximises the mean of their F1 values. The old benchmark is reused and the sealed slice is separate.'})
    pd.DataFrame(settings).to_csv(OUT/'tuning.csv',index=False)
    benchmark, sealed = add_features(benchmark,data),add_features(sealed,data)
    baseline = joblib.load(ROOT/'models/final/gradient_boosting.joblib')
    previous = json.loads((ROOT/'results/audience/training_selection.json').read_text())
    previous_selection = previous['selection']['imdb_show_only_local']
    prior_train = train_a.assign(imdb_show_votes=train_a.show_log_votes)
    encoder, prior_estimator = fit_selected('imdb_show_only_local',previous_selection,prior_train,{})
    tested = []
    for dataset, rows in [('course_benchmark',benchmark),('sealed_catalogue',sealed)]:
        values = baseline.predict_proba(rows[FEATURES])[:,1]
        tested.append({'regime':'final_baseline','dataset':dataset,**evaluate(rows,values,.5)})
        prior_rows = rows.assign(imdb_show_votes=rows.show_log_votes)
        values, _ = predict_audience(previous_selection,encoder,prior_estimator,prior_rows,{})
        add_evaluation(tested,'previous_imdb_show',dataset,rows,values,previous_selection['threshold'])
    variants = {row['regime']:row for row in best}
    if size_control != variants['expanded']:
        variants['expanded_same_model'] = size_control
    for regime, selected in variants.items():
        feature, method = selected['features'],selected['method']
        fitting = train_a if selected['regime']=='original' else pd.concat([train_a,train_b],ignore_index=True)
        model = fit(feature,method,fitting)
        joblib.dump((selected,model),OUT/f'{regime}_local.joblib',compress=3)
        for dataset,rows in [('course_benchmark',benchmark),('sealed_catalogue',sealed)]:
            values = predict(model,feature,method,rows)
            add_evaluation(tested,regime,dataset,rows,values,selected['threshold'])
    save_json(OUT/'evaluation.json',{'results':tested,'sealed_positives':int(sealed.high_rated.sum()),
              'sealed_rows':len(sealed),'sealed_imdb_coverage':int(sealed.imdb_show_rating.notna().sum()),
              'extra_positives':int(additional.high_rated.sum()),
              'scope':'Current cross-site ratings for already released shows. Ranking diagnostics use labels and are not measured deployment F1.'})
    print(pd.DataFrame(tested).to_string(index=False),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
