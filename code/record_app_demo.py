"""Record real HTTP-route outputs for the new-input presentation example."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app


def main():
    client = create_app().test_client()
    payload = {
        'premiere_year': 2026, 'runtime_minutes': 45, 'network': 'Unknown',
        'web_channel': 'Netflix', 'broadcast_country': 'US', 'distribution': 'Streaming',
        'genres': ['Drama', 'Mystery'],
        'plot_summary': 'An astronomer investigates a signal with a retired engineer.',
    }
    new = client.post('/api/predict', json=payload)
    assert new.status_code == 200, new.json
    missing = client.post('/api/predict', json={**payload, 'plot_summary': '', 'runtime_minutes': None})
    assert missing.status_code == 200 and missing.json['primary_role'] == 'metadata'
    invalid = client.post('/api/predict', json={**payload, 'runtime_minutes': -5})
    assert invalid.status_code == 400
    example = next(row for row in client.get('/api/examples').json if row['show_id'] == 2)
    example_prediction = client.post('/api/predict', json=example)
    assert example_prediction.status_code == 200
    record = {'input': payload, 'new_input': new.json, 'missing_text_and_runtime': missing.json,
              'invalid_runtime': {'status': invalid.status_code, **invalid.json},
              'snapshot_example': {'name': example['name'], 'observed_rating': example['observed_rating'],
                                   'prediction': example_prediction.json},
              'note': 'Fictional input has no known correct rating. Saved-model inference, no fitting.'}
    path = ROOT / 'results/final/app_demo.json'
    path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record['new_input'], indent=2))


if __name__ == '__main__':
    main()
