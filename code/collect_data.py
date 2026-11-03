"""Collect a dated, reproducible TVmaze show-index snapshot without authentication.

Example: python code/collect_data.py --snapshot midterm_2026-10-06 --pages 60
Existing raw pages are reused, and their original collection times are preserved.
Never recollect over the Midterm snapshot when extending the project.
"""
import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = 'AITU-ML-TVSeries-CourseProject/1.0 (educational snapshot)'


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(url):
    for attempt in range(5):
        req = urllib.request.Request(url, headers={
            'User-Agent': USER_AGENT, 'Accept': 'application/json,text/plain,text/html;q=0.8'})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read(), response.status
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(min(2 ** (attempt + 1), 20))
        except urllib.error.URLError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f'Repeated request failure: {url}')


def collect(snapshot, pages):
    directory = ROOT / 'data' / 'raw' / snapshot
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'collection_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'snapshot': snapshot, 'started_utc': now(), 'collector': 'code/collect_data.py',
        'user_agent': USER_AGENT, 'source': 'https://api.tvmaze.com/shows',
        'page_plan': list(range(pages)), 'sampling': 'All objects in sequential index pages 0 through the final planned page; pages chosen before modelling.',
        'request_interval_seconds': 0.65, 'pages': [],
    }
    if manifest['page_plan'] != list(range(pages)):
        raise ValueError('Use a new snapshot name for a different page plan.')
    existing = {record['page']: record for record in manifest['pages']}
    total = 0
    downloaded = False
    for page in range(pages):
        url = f'https://api.tvmaze.com/shows?page={page}'
        path = directory / f'page_{page:03d}.json'
        if page in existing and path.exists():
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != existing[page]['sha256']:
                raise ValueError(f'Changed raw snapshot: {path}')
        else:
            downloaded = True
            payload, status = fetch(url)
            records = json.loads(payload)
            if not isinstance(records, list):
                raise TypeError('Expected the TVmaze paginated show index.')
            path.write_bytes(payload)
            record = {'page': page, 'url': url, 'http_status': status,
                      'collected_utc': now(), 'rows': len(records),
                      'sha256': hashlib.sha256(payload).hexdigest(), 'file': path.name}
            manifest['pages'].append(record)
            manifest['pages'].sort(key=lambda x: x['page'])
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
            time.sleep(0.65)
        total += len(json.loads(payload))
        if page % 10 == 0 or page == pages - 1:
            print(f'Collected/reused {page + 1}/{pages} pages, {total:,} raw objects', flush=True)
    if downloaded or 'finished_utc' not in manifest:
        manifest['finished_utc'] = now()
    manifest.update({'raw_rows': total,
                     'collection_date_almaty': datetime.fromisoformat(manifest['finished_utc']).astimezone(ZoneInfo('Asia/Almaty')).date().isoformat()})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    check_path = directory / 'source_check.json'
    if not check_path.exists():
        checks = {'checked_utc': now(), 'authentication': 'No account, token or login used.',
                  'objects': 'TV show metadata and plot summaries. No user profiles or private personal records collected.',
                  'api_terms_url': 'https://www.tvmaze.com/api#licensing',
                  'data_license': 'CC BY-SA 4.0',
                  'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
                  'attribution': 'Source: TVmaze, https://www.tvmaze.com. API: https://www.tvmaze.com/api.',
                  'requests': []}
        for url in ['https://www.tvmaze.com/robots.txt', 'https://api.tvmaze.com/robots.txt',
                    'https://www.tvmaze.com/api']:
            try:
                payload, status = fetch(url)
                content = payload.decode('utf-8')
                record = {'url': url, 'http_status': status,
                          'sha256': hashlib.sha256(payload).hexdigest()}
                if url.endswith('robots.txt'):
                    record['text'] = content
                else:
                    match = re.search(r'Use of the TVmaze API.*?ShareAlike.*?</p>', content, re.S)
                    record['licensing_excerpt'] = re.sub('<[^>]+>', '', match.group(0)) if match else 'See licensing section at the recorded URL.'
                checks['requests'].append(record)
            except urllib.error.HTTPError as error:
                checks['requests'].append({'url': url, 'http_status': error.code,
                    'interpretation': 'No robots file at API host; use the expressly documented public API and its licensing/rate-limit conditions.'})
        checks['robots_interpretation'] = 'The website robots file restricts stats, offers and user-followed/watched paths. This collector calls the documented public show-index API and none of those paths.'
        check_path.write_text(json.dumps(checks, indent=2) + '\n')
    print('Saved raw snapshot and source checks:', directory, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default='midterm_2026-10-06')
    parser.add_argument('--pages', type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.pages <= 200:
        parser.error('Use between 1 and 200 pages for this educational snapshot.')
    collect(args.snapshot, args.pages)
