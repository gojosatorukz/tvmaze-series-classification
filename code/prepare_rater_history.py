"""Aggregate MovieLens ratings and prior rater behaviour without retaining user IDs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from prepare_audience_sources import CUTOFF, DATA, imdb_lookup, sha256
from project import ROOT

SOURCE = 'https://grouplens.org/datasets/movielens/32m/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--transfer', action='store_true', help='Prepare the fixed second catalogue slice')
    args = parser.parse_args()
    output = DATA/'transfer' if args.transfer else DATA
    shows = pd.read_csv(output/'cohort.csv' if args.transfer else ROOT/'data/processed/series_clean.csv')
    raw = [row for page in sorted(output.glob('page_*.json')) for row in json.loads(page.read_text())] if args.transfer else None
    ids = imdb_lookup(shows, raw)
    with ZipFile(args.archive) as dataset:
        with dataset.open('ml-32m/links.csv') as stream:
            ids_lens = pd.read_csv(stream, usecols=['movieId','imdbId'], dtype={'imdbId':str}).dropna()
        ids_lens['imdb'] = 'tt'+ids_lens.imdbId.str.zfill(7)
        ids_lens = ids_lens[~ids_lens.imdb.duplicated(keep=False)]
        mapping = dict(ids_lens.merge(ids, on='imdb',validate='one_to_one')[['movieId','show_id']].itertuples(index=False,name=None))
        print(f'{len(mapping)} exact IMDb ID matches in MovieLens links',flush=True)
        statistics = defaultdict(list)
        counts = {'matched_raters':0,'historical_profiles':0}
        cutoff = int(pd.Timestamp(CUTOFF,tz='UTC').timestamp())
        if mapping:
            with dataset.open('ml-32m/ratings.csv') as stream:
                reader = pd.read_csv(stream,chunksize=600_000,dtype={'userId':'int32','movieId':'int32','rating':'float32','timestamp':'int64'})
                trailing = None

                def process(chunk):
                    selected_users = chunk.loc[chunk.movieId.isin(mapping),'userId'].unique()
                    histories = chunk[chunk.userId.isin(selected_users)].sort_values(['userId','timestamp'])
                    for _, history in histories.groupby('userId',sort=False):
                        counts['matched_raters'] += 1
                        times = history.timestamp.to_numpy()
                        values = history.rating.to_numpy(dtype=float)
                        prefix = np.concatenate([[0.],np.cumsum(values)])
                        relevant = history[history.movieId.isin(mapping)]
                        for row in relevant.itertuples():
                            # Strictly past events only, with all ambiguous simultaneous events excluded.
                            before = int(np.searchsorted(times,row.timestamp,side='left'))
                            relative = row.rating - prefix[before]/before if before >= 10 else np.nan
                            counts['historical_profiles'] += int(before >= 10)
                            statistics[mapping[row.movieId]].append((row.rating,relative,before))

                for chunk in reader:
                    if trailing is not None:
                        chunk = pd.concat([trailing,chunk],ignore_index=True)
                    last = chunk.userId.iat[-1]
                    trailing = chunk[chunk.userId.eq(last)]
                    complete = chunk[chunk.userId.ne(last) & chunk.timestamp.lt(cutoff)]
                    process(complete)
                if trailing is not None:
                    process(trailing[trailing.timestamp.lt(cutoff)])
    features = []
    for show_id, events in statistics.items():
        values = np.asarray(events)
        relative = values[np.isfinite(values[:,1]),1]
        features.append({'show_id':show_id,'lens_votes':len(events),'lens_mean':values[:,0].mean(),
                         'lens_std':values[:,0].std(),'lens_relative_mean':relative.mean() if len(relative) else np.nan,
                         'lens_prior_count_mean':np.log1p(values[:,2]).mean(),
                         'lens_profile_share':len(relative)/len(events)})
    audit = {'source':SOURCE,'license': 'See https://files.grouplens.org/datasets/movielens/ml-32m-README.html',
             'archive_sha256':sha256(args.archive),'matched_series':len(features),
             'matched_events':sum(len(rows) for rows in statistics.values()),**counts,
             'cutoff':CUTOFF,'published_features':'Aggregates per series only. No user IDs or individual ratings.',
             'citation':'Harper and Konstan (2015), The MovieLens Datasets: History and Context, doi:10.1145/2827872.'}
    output.mkdir(parents=True,exist_ok=True)
    (output/'movielens_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if features:
        pd.DataFrame(features).sort_values('show_id').to_csv(output/'movielens_aggregate.csv',index=False)
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == '__main__':
    main()
