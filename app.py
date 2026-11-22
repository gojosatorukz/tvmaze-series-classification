"""Local TVmaze prediction app. Run with: python app.py"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from project import FEATURES, NUMERIC


@lru_cache(maxsize=1)
def experiment():
    return json.loads((ROOT / 'results/final/app_manifest.json').read_text())


@lru_cache(maxsize=16)
def saved_model(family):
    return joblib.load(ROOT / 'models/final' / f'{family}.joblib')


def model_input(payload):
    if not isinstance(payload, dict):
        raise ValueError('Enter a JSON object containing the series features.')
    record = {}
    for column, lower, upper in [('premiere_year', 1900, 2100), ('runtime_minutes', 1, 600)]:
        value = payload.get(column)
        if value is None or value == '':
            record[column] = np.nan
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{column.replace("_", " ").capitalize()} must be a number.') from None
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f'{column.replace("_", " ").capitalize()} must be between {lower} and {upper}.')
        if column == 'premiere_year' and not value.is_integer():
            raise ValueError('Premiere year must be a whole number.')
        record[column] = value
    for column in ['network', 'web_channel', 'broadcast_country', 'distribution']:
        default = 'None' if column == 'web_channel' else 'Unknown'
        value = payload.get(column, default)
        if value is None or value == '':
            value = default
        if not isinstance(value, str) or len(value) > 120:
            raise ValueError(f'{column.replace("_", " ").capitalize()} must be a short text value.')
        record[column] = value.strip() or default
    if record['distribution'] not in ['Broadcast', 'Streaming', 'Hybrid', 'Unknown']:
        raise ValueError('Choose Broadcast, Streaming, Hybrid or Unknown for distribution.')
    genres = payload.get('genres', [])
    if not isinstance(genres, list) or len(genres) > 20 or any(
            not isinstance(value, str) or len(value) > 60 for value in genres):
        raise ValueError('Genres must be a list of up to 20 short names.')
    record['genres'] = json.dumps(sorted({value.strip() for value in genres if value.strip()}))
    synopsis = payload.get('plot_summary', '')
    if not isinstance(synopsis, str) or len(synopsis) > 15000:
        raise ValueError('The synopsis must contain at most 15,000 characters.')
    record['plot_summary'] = synopsis.strip()
    return pd.DataFrame([record], columns=FEATURES + ['plot_summary'])


def create_app():
    app = Flask(__name__)
    app.json.sort_keys = False
    app.config['MAX_CONTENT_LENGTH'] = 64000

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.get('/api/experiment')
    def study():
        try:
            return jsonify(experiment())
        except FileNotFoundError:
            return jsonify(error='The experiment results are unavailable. Complete the documented setup first.'), 503

    @app.get('/api/examples')
    def examples():
        frame = pd.read_csv(ROOT / 'data/processed/series_clean.csv', keep_default_na=False)
        for column in NUMERIC:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
        selected = frame.loc[frame.show_id.isin([2, 5, 185, 318])].copy()
        records = []
        for row in selected.to_dict('records'):
            item = {key: row[key] for key in FEATURES + ['plot_summary', 'name', 'show_id']}
            item['genres'] = json.loads(item['genres'])
            item['observed_rating'] = float(row['rating'])
            for key in NUMERIC:
                if pd.isna(item[key]):
                    item[key] = None
            records.append(item)
        return jsonify(records)

    @app.post('/api/predict')
    def predict():
        try:
            frame = model_input(request.get_json(silent=True))
            manifest = experiment()
            has_text = bool(frame.iloc[0].plot_summary)
            results = []
            for role, specification in manifest['models'].items():
                if role != 'metadata' and not has_text:
                    results.append({'role': role, 'family': specification['family'], 'available': False})
                    continue
                model = saved_model(specification['family'])
                predicted = int(model.predict(frame)[0])
                if hasattr(model, 'predict_proba'):
                    positive = list(model.classes_).index(1)
                    value = float(model.predict_proba(frame)[0, positive])
                    score_kind, threshold = 'model score', .5
                else:
                    value = float(model.decision_function(frame)[0])
                    score_kind, threshold = 'SVM margin', 0
                results.append({'role': role, 'family': specification['family'], 'available': True,
                                'predicted': predicted, 'score': value, 'score_kind': score_kind,
                                'threshold': threshold, 'label': 'Rating at least 8' if predicted else 'Rating below 8'})
            primary = manifest['primary_role'] if has_text else 'metadata'
            return jsonify(models=results, primary_role=primary, has_text=has_text,
                           missing_numeric=[key for key in NUMERIC if pd.isna(frame.iloc[0][key])],
                           target='TVmaze rating ≥ 8/10', preprocessing='Saved, training-fitted pipelines')
        except (ValueError, TypeError) as error:
            return jsonify(error=str(error)), 400
        except FileNotFoundError:
            return jsonify(error='A saved model is unavailable. Follow the setup instructions.'), 503

    @app.errorhandler(413)
    def oversized(_error):
        return jsonify(error='The input is too large. Shorten the synopsis.'), 413

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    create_app().run(host='127.0.0.1', port=args.port, debug=False)
