"""Collect public TVmaze episode and production data for the frozen cohort."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/audience'
API = 'https://api.tvmaze.com'
SOURCE = 'https://www.tvmaze.com/api'
HEADERS = {'User-Agent': 'AITU-TVmaze-rating-study/2.0 (educational research)'}
_lock = threading.Lock()
_next_request = 0.0


def fetch_show(show_id):
    global _next_request
    response = None
    for attempt in range(5):
        with _lock:
            time.sleep(max(_next_request - time.monotonic(), 0))
            _next_request = time.monotonic() + .56
        try:
            response = requests.get(f'{API}/shows/{show_id}',
                                    params=[('embed[]', key) for key in ('episodes', 'cast', 'crew')],
                                    headers=HEADERS, timeout=(10, 30))
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            if response.status_code == 429:
                with _lock:
                    retry_after = response.headers.get('Retry-After', '')
                    delay = float(retry_after) if retry_after.isdigit() else max(5, 2**attempt)
                    _next_request = max(_next_request, time.monotonic() + delay)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 4:
                raise
        time.sleep(2**attempt)
    if response is None:
        raise RuntimeError(f'No TVmaze response for {show_id}')
    response.raise_for_status()
    raw = response.json()
    if raw['id'] != show_id or not isinstance(raw.get('_embedded'), dict):
        raise ValueError(f'Unexpected TVmaze show response for {show_id}')
    embedded = raw['_embedded']
    # Show-level rating, popularity weight and personal profile fields are never stored.
    return {
        'show_id': show_id, 'collected_utc': datetime.now(timezone.utc).isoformat(),
        'source_url': response.url, 'original_response_sha256': hashlib.sha256(response.content).hexdigest(),
        'episodes': [{**{key: episode.get(key) for key in ('id', 'season', 'number', 'airdate', 'runtime', 'summary')},
                      'episode_rating': (episode.get('rating') or {}).get('average')}
                     for episode in embedded.get('episodes', [])],
        'cast_person_ids': [item['person']['id'] for item in embedded.get('cast', [])],
        'crew_credits': [[item['type'], item['person']['id']] for item in embedded.get('crew', [])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', type=Path, default=ROOT / 'data/splits/split_assignments.csv')
    parser.add_argument('--output', type=Path, default=DATA / 'tvmaze_details.jsonl')
    args = parser.parse_args()
    ids = set(map(int, pd.read_csv(args.ids).show_id))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    known = set()
    if args.output.exists():
        with args.output.open() as source:
            known = {json.loads(line)['show_id'] for line in source if line.strip()}
    pending = sorted(ids - known)
    print(f'{len(ids)} requested, {len(ids & known)} cached, {len(pending)} to fetch. {SOURCE}', flush=True)
    failures = []

    def download(show_id):
        try:
            return show_id, fetch_show(show_id)
        except (requests.RequestException, KeyError, ValueError, RuntimeError) as error:
            return show_id, type(error).__name__ + ': ' + str(error)

    with ThreadPoolExecutor(max_workers=5) as pool, args.output.open('a', encoding='utf-8') as sink:
        for index, (show_id, result) in enumerate(pool.map(download, pending), 1):
            if isinstance(result, str):
                failures.append({'show_id': show_id, 'error': result})
            else:
                sink.write(json.dumps(result, ensure_ascii=False) + '\n')
                sink.flush()
            if index % 100 == 0 or index == len(pending):
                print(f'{index}/{len(pending)} processed, {len(failures)} failed', flush=True)
    args.output.with_suffix('.failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    if failures:
        print('Rerun with the same output path to retry missing IDs.', flush=True)


if __name__ == '__main__':
    main()
