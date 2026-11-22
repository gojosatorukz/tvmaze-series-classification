"""Freeze larger training and new validation catalogues for the IMDb follow-up."""
from datetime import datetime, timezone
import hashlib
import json
import time

import requests

from collect_audience import API, HEADERS
from prepare_audience_sources import CUTOFF, title_key
from project import ROOT, clean_raw, load_raw

OUTPUT = ROOT/'data/audience_scaleup'
TRAIN_PAGES = range(100, 200)
SEALED_PAGES = range(200, 240)


def load_page(number):
    path = OUTPUT/'index'/f'page_{number:03d}.json'
    url = f'{API}/shows?page={number}'
    if not path.exists():
        for attempt in range(5):
            response = requests.get(url,headers=HEADERS,timeout=(10,30))
            if response.status_code not in (429,500,502,503,504):
                break
            time.sleep(max(4,2**attempt))
        if response.status_code==404:
            raise ValueError(f'Planned index page {number} unavailable. Collection stopped.')
        response.raise_for_status()
        path.write_bytes(response.content)
        time.sleep(.56)
    raw = path.read_bytes()
    rows = json.loads(raw)
    return rows, {'page':number,'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'rows':len(rows)}


def keys(record):
    return ((record.get('externals') or {}).get('imdb'),
            (title_key(record['name']),(record.get('premiered') or '')[:4]))


def save_cohort(group, pages, seen_ids, seen_imdb, seen_names):
    records, manifests = [], []
    for page in pages:
        rows, metadata = load_page(page)
        manifests.append(metadata)
        for record in rows:
            value, name = keys(record)
            if record['id'] in seen_ids or value in seen_imdb or name in seen_names:
                continue
            records.append(record)
            seen_ids.add(record['id'])
            seen_names.add(name)
            if value:
                seen_imdb.add(value)
        if page%25==24 or page==pages.stop-1:
            print(f'{group}: index through page {page}, {len(records):,} distinct objects',flush=True)
    clean, _ = clean_raw(records,CUTOFF)
    path = OUTPUT/f'{group}.csv'
    clean.to_csv(path,index=False)
    manifest = {'pages':manifests,'rows':len(clean),'cohort_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'collected_utc':datetime.now(timezone.utc).isoformat(), 'premiere_cutoff':CUTOFF,
                'target':'TVmaze rating >= 8', 'source':'https://www.tvmaze.com/api'}
    (OUTPUT/f'{group}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{group}: froze {len(clean)} eligible series',flush=True)


def main():
    (OUTPUT/'index').mkdir(exist_ok=True,parents=True)
    original, _ = load_raw()
    transfer = ROOT/'data/audience/transfer'
    for path in sorted(transfer.glob('page_*.json')):
        original.extend(json.loads(path.read_text()))
    seen_ids = {row['id'] for row in original}
    seen_imdb = {value for row in original if (value:=keys(row)[0])}
    seen_names = {keys(row)[1] for row in original}
    save_cohort('training_extra',TRAIN_PAGES,seen_ids,seen_imdb,seen_names)
    save_cohort('sealed',SEALED_PAGES,seen_ids,seen_imdb,seen_names)


if __name__ == '__main__':
    main()
