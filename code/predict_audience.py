"""Run a saved reception model for a show in the bundled catalogue snapshot."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from audience_experiment import encode, predict
from audience_features import DATA, TEXT, build_frame
from audience_models import EXPERIMENTS, LENS
from project import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-id',type=int,required=True)
    parser.add_argument('--study',type=Path,default=ROOT/'results/audience')
    parser.add_argument('--model',choices=list(EXPERIMENTS))
    parser.add_argument('--local-imdb',type=Path)
    args = parser.parse_args()
    summary = json.loads((args.study/'training_selection.json').read_text())
    variant = args.model or summary['public_winner']
    if variant is None or variant not in summary['selection']:
        parser.error('Choose a model that exists in this study.')
    blocks = EXPERIMENTS[variant]
    if {'imdb','imdb_show'} & set(blocks) and args.local_imdb is None:
        parser.error('This local model needs the locally prepared IMDb input file.')
    cohort = pd.read_csv(ROOT/'data/processed/series_clean.csv',keep_default_na=False,
                         na_values={'runtime_minutes':[''],'premiere_year':['']})
    show = cohort[cohort.show_id.eq(args.show_id)]
    if show.empty:
        parser.error('The show ID is not in the bundled snapshot.')
    frame, _ = build_frame(show,DATA/'tvmaze_details.jsonl',DATA/'celluloid_excerpts.jsonl',args.local_imdb)
    if {'lens','lens_ratings'} & set(blocks):
        lens = pd.read_csv(DATA/'movielens_aggregate.csv').set_index('show_id',verify_integrity=True)
        frame = frame.join(lens[LENS],on='show_id')
    selected, encoder, model = joblib.load(args.study/'models'/f'{variant}.joblib')
    if selected != summary['selection'][variant]:
        raise ValueError(f'Model setting mismatch for {variant}')
    values, labels = predict(selected,encoder,model,frame,encode(frame,TEXT))
    print(json.dumps({'show_id':args.show_id,'title':show.name.iat[0],'variant':variant,
                      'prediction': 'TVmaze rating at least 8' if labels[0] else 'TVmaze rating below 8',
                      'score':float(values[0]),'threshold':selected['threshold'],
                      'first_three_episode_ratings_available':int(frame.first_three_rated.iat[0]),
                      'scope':'Retrospective inference with available sources. Score is uncalibrated.'},indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
