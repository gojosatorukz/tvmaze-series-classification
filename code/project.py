"""Shared cleaning, frozen splits and modelling functions for all course stages."""
import hashlib
import json
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = 'midterm_2026-10-06'
SEED = 42
THRESHOLD = 8.0
NUMERIC = ['premiere_year', 'runtime_minutes']
CATEGORICAL = ['network', 'web_channel', 'broadcast_country', 'distribution']
FEATURES = NUMERIC + CATEGORICAL + ['genres']


class SummaryParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)


def strip_html(text):
    parser = SummaryParser()
    parser.feed(text or '')
    return ' '.join(unescape(' '.join(parser.parts)).split())


def load_raw(root=ROOT):
    folder = root / 'data' / 'raw' / SNAPSHOT
    manifest = json.loads((folder / 'collection_manifest.json').read_text())
    objects = []
    for page in manifest['pages']:
        raw = (folder / page['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == page['sha256'], 'Raw snapshot hash mismatch'
        records = json.loads(raw)
        assert len(records) == page['rows']
        objects.extend(records)
    return objects, manifest


def clean_raw(objects, collection_date):
    audits = []
    frame = pd.DataFrame(objects)
    def keep(mask, reason):
        nonlocal frame
        before = len(frame)
        frame = frame.loc[mask].copy()
        audits.append({'step': reason, 'before': before, 'removed': before-len(frame), 'after': len(frame)})
    keep(~frame['id'].duplicated(keep='first'), 'Deduplicate identical show IDs')
    keep(frame['language'].eq('English'), 'English-language series only')
    keep(frame['type'].eq('Scripted'), 'Scripted series only')
    rating = frame['rating'].map(lambda x: x.get('average') if isinstance(x, dict) else None)
    keep(rating.notna() & rating.between(0, 10), 'Require a valid observed rating from 0 to 10')
    summary = frame['summary'].map(strip_html)
    keep(summary.str.len().gt(0), 'Require non-empty plot text for the future Final')
    premiere = pd.to_datetime(frame['premiered'], errors='coerce')
    keep(premiere.isna() | premiere.le(pd.Timestamp(collection_date)), 'Exclude premieres after collection date')
    result = []
    for record in frame.to_dict('records'):
        network = record.get('network') or {}
        channel = record.get('webChannel') or {}
        country = network.get('country') or channel.get('country') or {}
        premiered = pd.to_datetime(record.get('premiered'), errors='coerce')
        runtime = pd.to_numeric(record.get('runtime'), errors='coerce')
        if pd.isna(runtime) or runtime <= 0:
            runtime = np.nan
        rating = float(record['rating']['average'])
        result.append({
            'show_id': int(record['id']), 'name': record['name'], 'source_url': record['url'],
            'language': record['language'], 'show_type': record['type'],
            'premiered': record.get('premiered'),
            'premiere_year': float(premiered.year) if pd.notna(premiered) else np.nan,
            'runtime_minutes': runtime,
            'network': network.get('name') or 'Unknown',
            'web_channel': channel.get('name') or 'None',
            'broadcast_country': country.get('code') or 'Unknown',
            'distribution': ('Hybrid' if network and channel else 'Broadcast' if network else 'Streaming' if channel else 'Unknown'),
            'genres': json.dumps(sorted(set(record.get('genres') or []))),
            'rating': rating, 'high_rated': int(rating >= THRESHOLD),
            'plot_summary': strip_html(record.get('summary')),
        })
    clean = pd.DataFrame(result).sort_values('show_id').reset_index(drop=True)
    return clean, pd.DataFrame(audits)


def freeze_splits(clean, root=ROOT):
    folder = root / 'data' / 'splits'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'split_assignments.csv'
    fingerprint = hashlib.sha256(clean[['show_id','high_rated']].to_csv(index=False).encode()).hexdigest()
    if path.exists():
        assignments = pd.read_csv(path)
        meta = json.loads((folder/'split_manifest.json').read_text())
        assert meta['id_label_sha256'] == fingerprint, 'Do not silently change the frozen cohort or labels.'
        assert set(assignments.show_id) == set(clean.show_id)
    else:
        train_ids, test_ids = train_test_split(clean.show_id, test_size=.20, random_state=SEED,
                                             stratify=clean.high_rated)
        assignments = clean[['show_id']].copy()
        assignments['split'] = np.where(assignments.show_id.isin(set(train_ids)), 'train', 'test')
        assignments['cv_fold'] = -1
        train = clean[clean.show_id.isin(set(train_ids))].reset_index(drop=True)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for fold, (_, validation) in enumerate(cv.split(train, train.high_rated), start=1):
            ids = train.iloc[validation].show_id
            assignments.loc[assignments.show_id.isin(ids), 'cv_fold'] = fold
        assignments.to_csv(path, index=False)
        (folder/'split_manifest.json').write_text(json.dumps({
            'seed': SEED, 'test_fraction': .20, 'stratified': True, 'cv_folds': 5,
            'target_threshold': THRESHOLD, 'id_label_sha256': fingerprint,
            'train_rows': len(train_ids), 'test_rows': len(test_ids),
            'policy': 'Reuse these exact show IDs and labels at Endterm and Final. Keep the Midterm snapshot immutable.'
        }, indent=2)+'\n')
    labelled = clean.merge(assignments, on='show_id', validate='one_to_one')
    train = labelled[labelled.split.eq('train')].reset_index(drop=True)
    test = labelled[labelled.split.eq('test')].reset_index(drop=True)
    folds = [(np.flatnonzero(train.cv_fold.to_numpy()!=fold),
              np.flatnonzero(train.cv_fold.to_numpy()==fold)) for fold in range(1,6)]
    assert set(train.show_id).isdisjoint(test.show_id)
    assert all(len(set(train.iloc[a].show_id) & set(train.iloc[b].show_id)) == 0 for a,b in folds)
    return train, test, folds


class GenreEncoder(TransformerMixin, BaseEstimator):
    """Learn genre vocabulary on training rows only; safely ignore unseen genres."""
    def fit(self, X, y=None):
        self.encoder_ = MultiLabelBinarizer()
        self.encoder_.fit([json.loads(value) for value in np.asarray(X).ravel()])
        self.classes_ = self.encoder_.classes_
        self.n_features_in_ = 1
        return self
    def transform(self, X):
        known = set(self.classes_)
        return self.encoder_.transform([[g for g in json.loads(value) if g in known]
                                       for value in np.asarray(X).ravel()])
    def get_feature_names_out(self, input_features=None):
        return np.array(['genre_'+value for value in self.classes_], dtype=object)


def preprocessing(scale):
    steps = [('impute', SimpleImputer(strategy='median', add_indicator=True))]
    if scale:
        steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(steps), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                     min_frequency=10, sparse_output=False), CATEGORICAL),
        ('genres', GenreEncoder(), ['genres']),
    ], remainder='drop', sparse_threshold=0)


def make_models():
    return {
        'Dummy majority': Pipeline([('prepare', preprocessing(False)),
                                    ('model', DummyClassifier(strategy='most_frequent'))]),
        'Dummy stratified': Pipeline([('prepare', preprocessing(False)),
                                      ('model', DummyClassifier(strategy='stratified', random_state=SEED))]),
        'Decision Tree': Pipeline([('prepare', preprocessing(False)),
                                  ('model', DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                                  class_weight='balanced', random_state=SEED))]),
        'KNN': Pipeline([('prepare', preprocessing(True)),
                        ('model', KNeighborsClassifier(n_neighbors=15, weights='distance', p=2))]),
        'SVM': Pipeline([('prepare', preprocessing(True)),
                        ('model', SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'))]),
    }


def positive_scores(model, X):
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(X))
    return model.predict_proba(X)[:, 1]
