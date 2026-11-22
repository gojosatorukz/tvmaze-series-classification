"""Use frozen model choices on a disjoint set of TVmaze catalogue entries."""
import argparse
import json
from pathlib import Path

from audience_experiment import encode, predict
from audience_features import DATA, build_frame
from audience_models import EXPERIMENTS, LENS
from endterm import scores
from final_experiments import save_json
from prepare_audience_sources import sha256
from project import FEATURES, ROOT

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'results/audience')
    parser.add_argument('--slice',type=Path,default=DATA/'transfer')
    parser.add_argument('--local-imdb',type=Path)
    args = parser.parse_args()
    cohort_path = args.slice/'cohort.csv'
    manifest = json.loads((args.slice/'manifest.json').read_text())
    if sha256(cohort_path) != manifest['sha256']:
        raise ValueError('Transfer cohort hash changed')
    cohort = pd.read_csv(cohort_path,keep_default_na=False,na_values={'runtime_minutes':[''],'premiere_year':['']})
    selection = json.loads((args.study/'training_selection.json').read_text())
    # Apply all frozen variants. The separate slice supplies a descriptive check for each row.
    winners = [name for name in selection['selection'] if args.local_imdb or not name.endswith('_local')]
    blocks = {block for name in winners for block in EXPERIMENTS[name]}
    frame, coverage = build_frame(cohort,args.slice/'tvmaze_details.jsonl',args.slice/'celluloid_excerpts.jsonl',args.local_imdb)
    if coverage['matched_details'] != len(frame):
        raise ValueError(f'Transfer details incomplete: {coverage["matched_details"]}/{len(frame)}')
    if {'lens','lens_ratings'} & blocks:
        path = args.slice/'movielens_aggregate.csv'
        if path.exists():
            frame = frame.join(pd.read_csv(path).set_index('show_id',verify_integrity=True)[LENS],on='show_id')
        else:
            frame[LENS] = float('nan')
        coverage['movielens_matched'] = int(frame.lens_votes.notna().sum())
    vectors = encode(frame,['plot_summary','episode_text','review_text'])
    original = joblib.load(ROOT/'models/final/gradient_boosting.joblib')
    results = [{'experiment':'final_gradient_boosting',**scores(frame.high_rated,
                original.predict(frame[FEATURES]),original.predict_proba(frame[FEATURES])[:,1])}]
    for name in winners:
        setting, encoder, model = joblib.load(args.study/'models'/f'{name}.joblib')
        if setting != selection['selection'][name]:
            raise ValueError(f'Saved setting mismatch for {name}')
        values, labelled = predict(setting,encoder,model,frame,vectors)
        results.append({'experiment':name,**scores(frame.high_rated,labelled,values)})
    save_json(args.study/'transfer_metrics.json',{'cohort_sha256':manifest['sha256'],'rows':len(frame),
              'positives':int(frame.high_rated.sum()),'coverage':coverage,'metrics':results,
              'scope':'Different TVmaze catalogue IDs. Current audience signals and current show ratings. No prospective voting history.'})
    print(pd.DataFrame(results)[['experiment','f1','precision','recall','average_precision']].to_string(index=False),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
