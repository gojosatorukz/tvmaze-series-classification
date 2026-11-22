"""Audit exact cross-source matches and prepare optional audience inputs."""
import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

from project import ROOT, load_raw

DATA = ROOT / 'data/audience'
CUTOFF = '2026-10-06'
CELLULOID = 'https://celluloidscore.com'
TWEETINGS_REVISION = '44c525d0c766944910686c60697203cda39305d6'
SCORE_PATTERN = re.compile(
    r'(?i)\b\d+(?:\.\d+)?\s*(?:/\s*\d+|out of\s+\d+|%|stars?\b)|'
    r'\b(?:TVmaze|IMDb|Metacritic|Metascore|Letterboxd|Rotten Tomatoes|Celluloid Score|Tomatometer)\b')
INSTRUCTIONS = re.compile(r'(?i)ignore (?:all |the )?(?:previous|prior|above) instructions|'
                          r'(?:reveal|print|show) (?:the )?system prompt|you are (?:chatgpt|codex)|'
                          r'<\|(?:system|assistant)\|>')


def title_key(title):
    return ''.join(character for character in unicodedata.normalize('NFKC', title).casefold()
                   if character.isalnum())


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def imdb_lookup(shows, raw=None):
    if raw is None:
        raw, _ = load_raw()
    crosswalk = {row['id']: (row.get('externals') or {}).get('imdb') for row in raw}
    ids = shows[['show_id']].assign(imdb=shows.show_id.map(crosswalk)).dropna()
    return ids[~ids.imdb.duplicated(keep=False)]


def prepare_reviews(shows, index_path, text_path):
    index = json.loads(index_path.read_text())
    shows_by_key = defaultdict(list)
    for row in shows.dropna(subset=['premiere_year']).itertuples():
        shows_by_key[(title_key(row.name), int(row.premiere_year))].append(row.show_id)
    index_shows = defaultdict(list)
    index_reviews = {}
    for row in index:
        metadata = row.get('meta') or ''
        year = re.match(r'^((?:19|20)\d{2})\b', metadata)
        if row['type'] == 'show' and year:
            index_shows[(title_key(row['title']), int(year[1]))].append(row)
        if row['type'] == 'review' and metadata.startswith('English · By '):
            index_reviews[CELLULOID+row['url']] = metadata.split(' · ')[1].removeprefix('By ')
    show_by_review = {}
    for key, candidates in index_shows.items():
        if len(candidates) == 1 and len(shows_by_key[key]) == 1:
            match = candidates[0]
            url = f'{CELLULOID}/reviews/{match["slug"]}-review/'
            if url in index_reviews:
                show_by_review[url] = (shows_by_key[key][0], CELLULOID+match['url'])
    passages = defaultdict(list)
    suspicious, removed, dated_after = [], 0, set()
    with text_path.open() as source:
        for line in source:
            item = json.loads(line)
            url = item['sourceUrl']
            if url not in show_by_review or item['contentType'] != 'review':
                continue
            updated = item.get('lastUpdated', '')
            try:
                eligible = date.fromisoformat(updated) <= date.fromisoformat(CUTOFF)
            except (TypeError, ValueError):
                eligible = False
            if not eligible:
                dated_after.add(url)
                continue
            if item['sectionHeading'] != 'Review excerpt':
                continue
            prefix = f' — {item["sectionHeading"]}. '
            _, separator, body = item['text'].partition(prefix)
            if not separator:
                continue
            for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z“"\'])', body):
                commands = list(INSTRUCTIONS.finditer(sentence))
                if commands:
                    suspicious.extend({'source': url, 'exact_text': command.group()} for command in commands)
                    continue
                if SCORE_PATTERN.search(sentence):
                    removed += 1
                    continue
                passages[url].append((updated, sentence))
    records = []
    for url, entries in sorted(passages.items()):
        text = ' '.join(dict.fromkeys(value for _, value in entries))
        if not text.strip():
            continue
        ident, show_url = show_by_review[url]
        records.append({'show_id': int(ident), 'review_updated': max(day for day, _ in entries),
                        'review_url': url, 'title_url': show_url, 'byline': index_reviews[url],
                        'review_text': text})
    audit = {'unique_title_year_review_candidates': len(show_by_review),
             'usable_series': len(records), 'later_or_undated_reviews': len(dated_after),
             'removed_score_sentences': removed, 'suspicious_passages': suspicious,
             'byline_counts': dict(Counter(row['byline'] for row in records)),
             'source': f'{CELLULOID}/about/data/', 'license': 'CC BY 4.0',
             'input_sha256': {path.name: sha256(path) for path in (index_path, text_path)}}
    return records, audit


def audit_raters(ids, path):
    wanted = set(ids.imdb)
    matched_shows, matched_raters = set(), set()
    count = 0
    with path.open() as handle:
        for line in handle:
            user, movie, _, timestamp = line.strip().split('::')
            imdb_id = 'tt'+movie.zfill(7)
            if imdb_id in wanted and int(timestamp) < int(pd.Timestamp(CUTOFF, tz='UTC').timestamp()):
                count += 1
                matched_shows.add(imdb_id)
                matched_raters.add(user)
    return {'source': f'https://github.com/sidooms/MovieTweetings/tree/{TWEETINGS_REVISION}',
            'matched_series': len(matched_shows), 'matched_events': count,
            'distinct_pseudonymous_raters': len(matched_raters), 'input_sha256': sha256(path),
            'account_mapping_downloaded': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path, help='Directory with locally downloaded public datasets')
    parser.add_argument('--transfer', action='store_true', help='Prepare the fixed second catalogue slice')
    args = parser.parse_args()
    sources = args.sources
    output = DATA/'transfer' if args.transfer else DATA
    output.mkdir(exist_ok=True, parents=True)
    shows = pd.read_csv(output/'cohort.csv' if args.transfer else ROOT/'data/processed/series_clean.csv')
    raw = [row for page in sorted(output.glob('page_*.json')) for row in json.loads(page.read_text())] if args.transfer else None
    ids = imdb_lookup(shows, raw)
    reviews, review_audit = prepare_reviews(shows, sources/'celluloid-index.json', sources/'celluloid-chunks.jsonl')
    with (output/'celluloid_excerpts.jsonl').open('w', encoding='utf-8') as stream:
        for row in reviews:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    raters = audit_raters(ids, sources/'movietweetings-ratings.dat')
    audit = {'cohort_rows': len(shows), 'unique_imdb_matches_possible': len(ids),
             'celluloid': review_audit, 'movietweetings': raters, 'label_date': CUTOFF}
    imdb_file = sources/'imdb-title-ratings.tsv.gz'
    if imdb_file.exists():
        ratings = pd.read_csv(imdb_file, sep='\t', usecols=['tconst','averageRating','numVotes'])
        local = ids.merge(ratings, left_on='imdb', right_on='tconst', validate='one_to_one')
        local = local[['show_id','averageRating','numVotes']].rename(columns={
            'averageRating':'imdb_show_rating','numVotes':'imdb_show_votes'})
        audit['imdb_local'] = {'matched_series': len(local), 'source': 'https://data.imdb.com/non-commercial-datasets/',
                               'input_sha256': sha256(imdb_file), 'scope': 'Personal non-commercial analysis. Keep matched ratings local.'}
        episode_file = sources/'imdb-title-episode.tsv.gz'
        if episode_file.exists():
            episodes = pd.read_csv(episode_file, sep='\t', na_values='\\N')
            episodes = episodes.merge(ids, left_on='parentTconst', right_on='imdb')
            episodes = episodes[episodes.seasonNumber.gt(0) & episodes.episodeNumber.gt(0)]
            counts = episodes.groupby(['show_id','seasonNumber','episodeNumber']).tconst.transform('size')
            episodes = episodes[counts.eq(1)].merge(ratings, on='tconst')
            early = episodes[(episodes.seasonNumber == 1) & episodes.episodeNumber.between(1, 3)]
            features = early.groupby('show_id').agg(imdb_ep_rating=('averageRating','mean'),
                                                    imdb_ep_count=('tconst','size'),
                                                    imdb_ep_votes=('numVotes','sum')).reset_index()
            local = local.merge(features, on='show_id', how='left', validate='one_to_one')
            all_features = episodes.assign(high=episodes.averageRating.ge(8)).groupby('show_id').agg(
                imdb_all_rating=('averageRating','mean'),imdb_all_ep_count=('tconst','size'),
                imdb_all_ep_votes=('numVotes','sum'),imdb_all_share_high=('high','mean')).reset_index()
            local = local.merge(all_features,on='show_id',how='left',validate='one_to_one')
            audit['imdb_local']['first_three_episode_matched_series'] = len(features)
            audit['imdb_local']['episode_input_sha256'] = sha256(episode_file)
        local.to_csv(sources/('imdb-local-transfer.csv' if args.transfer else 'imdb-local-matched.csv'), index=False)
    (output/'source_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
