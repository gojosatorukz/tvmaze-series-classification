"""Freeze a new disjoint, fixed-page TVmaze catalogue slice for a transfer check."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import pandas as pd
import requests

from collect_audience import API, HEADERS
from prepare_audience_sources import CUTOFF
from project import ROOT, clean_raw, load_raw

PAGES = range(60, 100)
DEST = ROOT/'data/audience/transfer'


def imdb(row):
    return (row.get('externals') or {}).get('imdb')


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    objects, pages = [], []
    for page in PAGES:
        url = f'{API}/shows?page={page}'
        path = DEST/f'page_{page:03d}.json'
        if not path.exists():
            for attempt in range(5):
                response = requests.get(url,headers=HEADERS,timeout=(10,30))
                if response.status_code not in (429,500,502,503,504):
                    break
                time.sleep(max(6,2**attempt))
            response.raise_for_status()
            path.write_bytes(response.content)
            # Allow the primary catalogue collector to continue within the same shared rate limit.
            time.sleep(6.5)
        raw = path.read_bytes()
        rows = json.loads(raw)
        objects.extend(rows)
        pages.append({'page':page,'url':url,'rows':len(rows),'sha256':hashlib.sha256(raw).hexdigest()})
    original, _ = load_raw()
    ids = {row['id'] for row in original}
    foreign = {imdb(row) for row in original if imdb(row)}
    names = {(re.sub(r'\W+','',row['name'].casefold()),(row.get('premiered') or '')[:4]) for row in original}
    combined = [row for row in objects if row['id'] not in ids and imdb(row) not in foreign and
                (re.sub(r'\W+','',row['name'].casefold()),(row.get('premiered') or '')[:4]) not in names]
    cohort, _ = clean_raw(combined,CUTOFF)
    path = DEST/'cohort.csv'
    cohort.to_csv(path,index=False)
    out = {'selection': 'Fixed following show index pages 60–99. The initial ten pages gave 76 eligible shows, so the slice was extended using coverage only, before evaluating any model. Original raw TVmaze IDs, IMDb IDs and exact normalized title-year matches excluded. Same label definition and eligibility with premieres no later than the original cutoff.',
           'collected_utc':datetime.now(timezone.utc).isoformat(),'pages':pages,'series':len(cohort),
           'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'role':'Separate catalogue transfer check. Do not use it for selection or threshold tuning.'}
    (DEST/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f'Frozen fresh catalogue: {len(cohort)} unique eligible shows',flush=True)


if __name__ == '__main__':
    main()
