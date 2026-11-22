"""Construct disjoint catalogue, episode, review and optional IMDb feature blocks."""
from collections import defaultdict
from datetime import date
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from prepare_audience_sources import CUTOFF, INSTRUCTIONS, SCORE_PATTERN
from project import FEATURES, ROOT, strip_html

DATA = ROOT / 'data/audience'
STRUCTURE = ['crew_count','cast_count','season_one_count','first_gap','episode_runtime',
             'episode_runtime_sd','premiere_month','synopsis_words']
REACTION = ['first_three_rated','pilot_rating','first_three_mean','first_three_min',
            'first_three_max','first_three_sd','first_three_slope','first_three_share_high']
SEASON = ['season_one_rated','season_one_mean','season_one_sd','season_one_share_high']
ALL_EPISODES = ['all_episodes_rated','all_episodes_mean','all_episodes_median','all_episodes_sd',
                'all_episodes_share_high','all_episodes_q10','all_episodes_q90','all_episodes_recent_mean',
                'all_episodes_rating_coverage']
REVIEWS = ['has_review','review_words']
IMDB = ['imdb_show_rating','imdb_show_votes','imdb_ep_rating','imdb_ep_count','imdb_ep_votes']
IMDB_ALL = ['imdb_all_rating','imdb_all_ep_count','imdb_all_ep_votes','imdb_all_share_high']
TEXT = ['plot_summary','episode_text','review_text']


def eligible_episodes(record):
    valid = [episode for episode in record['episodes'] if type(episode['season']) is int
             and episode['season'] > 0 and type(episode['number']) is int and episode['number'] > 0
             and isinstance(episode['airdate'], str) and episode['airdate'] <= CUTOFF]
    return sorted(valid, key=lambda ep: (ep['season'], ep['number'], ep['airdate'], ep['id']))


def distinct_episodes(episodes):
    positions = defaultdict(list)
    for episode in episodes:
        positions[episode['season'], episode['number']].append(episode)
    return [rows[0] for rows in positions.values() if len(rows) == 1]


def episode_values(episodes):
    values = [(ep['number'], float(ep['episode_rating'])) for ep in episodes
              if type(ep['episode_rating']) in (float, int) and 1 <= ep['episode_rating'] <= 10]
    return dict(values)


def single_record(show, detail, review, suspicious):
    all_episodes = distinct_episodes(eligible_episodes(detail)) if detail else []
    first_season = [ep for ep in all_episodes if ep['season']==1]
    early = [ep for ep in first_season if ep['number'] <= 3]
    three = episode_values(early)
    season = np.asarray(list(episode_values(first_season).values()))
    complete = np.asarray([float(ep['episode_rating']) for ep in all_episodes
                           if type(ep['episode_rating']) in (float,int) and 1 <= ep['episode_rating'] <= 10])
    ratings = np.asarray(list(three.values()))
    cast = set(detail['cast_person_ids']) if detail else set()
    crew = {f'{kind}:{ident}' for kind, ident in detail['crew_credits']
            if kind in ('Creator','Executive Producer','Director','Writer','Developer')} if detail else set()
    word_blocks = []
    for episode in early:
        value = strip_html(episode.get('summary'))
        commands = list(INSTRUCTIONS.finditer(value))
        if commands:
            suspicious.extend({'source': f'https://api.tvmaze.com/episodes/{episode["id"]}',
                               'exact_text': command.group()} for command in commands)
        elif value and not SCORE_PATTERN.search(value):
            word_blocks.append(value)
    runtimes = [episode['runtime'] for episode in early if (episode['runtime'] or 0) > 0]
    premiere = pd.to_datetime(show['premiered'], errors='coerce')
    review_text = review.get('review_text','') if review else ''
    return {
        'crew_count':len(crew) if detail else np.nan, 'cast_count':len(cast) if detail else np.nan,
        'season_one_count':len(first_season) if detail else np.nan,
        'first_gap':(date.fromisoformat(early[1]['airdate'])-date.fromisoformat(early[0]['airdate'])).days if len(early)>=2 else np.nan,
        'episode_runtime':float(np.mean(runtimes)) if runtimes else np.nan,
        'episode_runtime_sd':float(np.std(runtimes)) if runtimes else np.nan,
        'premiere_month':premiere.month, 'synopsis_words':len(show['plot_summary'].split()),
        'first_three_rated':len(three), 'pilot_rating':three.get(1, np.nan),
        'first_three_mean':ratings.mean() if ratings.size else np.nan,
        'first_three_min':ratings.min() if ratings.size else np.nan,
        'first_three_max':ratings.max() if ratings.size else np.nan,
        'first_three_sd':ratings.std() if ratings.size >= 2 else np.nan,
        'first_three_slope':(three[max(three)]-three[min(three)])/(max(three)-min(three)) if len(three)>=2 else np.nan,
        'first_three_share_high':np.mean(ratings >= 8.) if ratings.size else np.nan,
        'season_one_rated':len(season), 'season_one_mean':season.mean() if season.size else np.nan,
        'season_one_sd':season.std() if season.size >= 2 else np.nan,
        'season_one_share_high':np.mean(season >= 8.) if season.size else np.nan,
        'all_episodes_rated':int(complete.size),
        'all_episodes_mean':complete.mean() if complete.size else np.nan,
        'all_episodes_median':np.median(complete) if complete.size else np.nan,
        'all_episodes_sd':complete.std() if complete.size >= 2 else np.nan,
        'all_episodes_share_high':np.mean(complete >= 8.) if complete.size else np.nan,
        'all_episodes_q10':np.quantile(complete,.1) if complete.size else np.nan,
        'all_episodes_q90':np.quantile(complete,.9) if complete.size else np.nan,
        'all_episodes_recent_mean':complete[-max(1, len(complete)//4):].mean() if complete.size else np.nan,
        'all_episodes_rating_coverage':len(complete)/len(all_episodes) if all_episodes else np.nan,
        'has_review':int(bool(review_text)), 'review_words':len(review_text.split()),
        'episode_text':' '.join(word_blocks), 'review_text':review_text,
        'credits':tuple(sorted([f'cast:{ident}' for ident in cast] + list(crew))),
    }


def build_frame(shows, details_path, reviews_path, imdb_path=None):
    def read_items(path):
        if path is None:
            return {}
        with Path(path).open() as stream:
            items = [json.loads(line) for line in stream if line.strip()]
        by_id = {row['show_id']:row for row in items}
        if len(by_id) != len(items):
            raise ValueError(f'Duplicate show IDs in {path}')
        return by_id

    details, reviews = read_items(details_path), read_items(reviews_path)
    suspicious = []
    output = pd.DataFrame([single_record(show, details.get(show['show_id']), reviews.get(show['show_id']), suspicious)
                           for show in shows.to_dict('records')], index=shows.index)
    frame = shows[['show_id', *FEATURES, 'plot_summary','high_rated','rating']].join(output)
    if imdb_path:
        local = pd.read_csv(imdb_path).set_index('show_id', verify_integrity=True)
        frame = frame.join(local, on='show_id')
        for col in IMDB+IMDB_ALL:
            if col not in frame:
                frame[col] = np.nan
        for col in ('imdb_show_votes','imdb_ep_votes','imdb_all_ep_votes'):
            frame[col] = np.log1p(frame[col])
    return frame, {
        'matched_details':len(set(shows.show_id) & details.keys()),
        'matched_reviews':int(output.has_review.sum()),
        'has_episode_text':int(output.episode_text.ne('').sum()),
        'has_first_three_ratings':int(output.first_three_rated.gt(0).sum()),
        'has_season_one_ratings':int(output.season_one_rated.gt(0).sum()),
        'suspicious_episode_passages':suspicious,
        'collected_utc_range':[min((row['collected_utc'] for row in details.values()),default=None),
                                max((row['collected_utc'] for row in details.values()),default=None)],
    }
