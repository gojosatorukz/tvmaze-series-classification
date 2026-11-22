"""Summarise paired model changes and inspect real errors on the fixed test IDs."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from audience_experiment import encode, predict
from audience_features import ALL_EPISODES, DATA, REACTION, SEASON, TEXT, build_frame
from audience_models import EXPERIMENTS, LENS
from endterm import check_continuity
from final_analysis import bootstrap_delta
from final_experiments import save_json
from project import ROOT


def main():
    study = ROOT/'results/audience'
    selection = json.loads((study/'training_selection.json').read_text())
    primary = selection['public_winner']
    expected = pd.read_csv(study/'benchmark_predictions.csv')
    actual = expected.high_rated.to_numpy()
    pairs = [('catalogue','production'), ('production','first_three'), ('first_three','first_season'),
             ('first_season','all_episodes'), ('first_season','with_reviews'),
             ('with_reviews','with_lens_ratings'), ('with_lens_ratings','with_rater_history'),
             ('imdb_show_only_local','imdb_only_local'), ('imdb_only_local','cross_site_local')]
    rows = []
    for left,right in pairs:
        if left not in expected or right not in expected:
            continue
        a,b = expected[left].to_numpy(),expected[right].to_numpy()
        rows.append({'base':left,'additional_input':right, 'f1_difference':f1_score(actual,b)-f1_score(actual,a),
                     'repaired':int(np.sum((a!=actual) & (b==actual))),
                     'introduced':int(np.sum((a==actual) & (b!=actual))),
                     **bootstrap_delta(actual,a,b)})
    pd.DataFrame(rows).to_csv(study/'paired_ablation.csv',index=False)
    _, _, test, _, _, _ = check_continuity(ROOT)
    frame,_ = build_frame(test,DATA/'tvmaze_details.jsonl',DATA/'celluloid_excerpts.jsonl')
    np.testing.assert_array_equal(frame.show_id,expected.show_id)
    if {'lens','lens_ratings'} & set(EXPERIMENTS[primary]):
        lens = pd.read_csv(DATA/'movielens_aggregate.csv').set_index('show_id',verify_integrity=True)
        frame = frame.join(lens[LENS],on='show_id')
    setting, encoder, fitted = joblib.load(study/'models'/f'{primary}.joblib')
    old = expected.final_gradient_boosting.to_numpy()
    new = expected[primary].to_numpy()
    examples = []
    for kind,indices in [('repaired false negative',np.flatnonzero((old==0) & (actual==1) & (new==1))),
                         ('repaired false positive',np.flatnonzero((old==1) & (actual==0) & (new==0))),
                         ('new false negative',np.flatnonzero((old==1) & (actual==1) & (new==0))),
                         ('new false positive',np.flatnonzero((old==0) & (actual==0) & (new==1)))]:
        if not len(indices):
            continue
        index = next((i for i in indices if frame.first_three_rated.iloc[i]>0),indices[0])
        raw = frame.iloc[index]
        masked = frame.iloc[[index]].copy()
        masked[REACTION+SEASON+ALL_EPISODES] = float('nan')
        masked[['first_three_rated','season_one_rated','all_episodes_rated','all_episodes_rating_coverage']] = 0
        no_ep_scores,_ = predict(setting,encoder,fitted,masked,encode(masked,TEXT))
        examples.append({'kind':kind,'show_id':int(raw.show_id),'name':test.name.iloc[index],
                         'true_rating':float(raw.rating),'new_score':float(expected[primary+'_score'].iloc[index]),
                         'score_with_episode_ratings_masked':float(no_ep_scores[0]),
                         'first_three_mean':float(raw.first_three_mean),'season_one_mean':float(raw.season_one_mean),
                         'all_episode_mean':float(raw.all_episodes_mean),'rated_episodes':int(raw.all_episodes_rated),
                         'interpretation':'Mask all episode ratings while keeping other inputs fixed. This measures sensitivity of the fitted model.'})
    save_json(study/'audience_error_examples.json',examples)
    print('Saved paired source comparisons and concrete series examples.')


if __name__ == '__main__':
    main()
