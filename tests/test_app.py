"""Exercise the submitted app through its normal HTTP entry points."""
import os
from pathlib import Path
import sys
import unittest

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app


class PredictionAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()
        cls.payload = {
            'premiere_year': 2020, 'runtime_minutes': 45, 'network': 'Unknown',
            'web_channel': 'Netflix', 'broadcast_country': 'US', 'distribution': 'Streaming',
            'genres': ['Drama', 'Crime'],
            'plot_summary': 'A detective investigates a disappearance in a small coastal town.',
        }

    def test_new_input_uses_three_saved_models(self):
        result = self.client.post('/api/predict', json=self.payload)
        self.assertEqual(result.status_code, 200, result.get_data(as_text=True))
        self.assertEqual({row['role'] for row in result.json['models']}, {'metadata', 'text', 'combined'})
        self.assertTrue(all(row['available'] for row in result.json['models']))
        self.assertTrue(all(row['predicted'] in [0, 1] for row in result.json['models']))

    def test_missing_synopsis_returns_metadata_prediction(self):
        payload = {**self.payload, 'plot_summary': '', 'runtime_minutes': None}
        result = self.client.post('/api/predict', json=payload)
        self.assertEqual(result.status_code, 200)
        by_role = {row['role']: row for row in result.json['models']}
        self.assertTrue(by_role['metadata']['available'])
        self.assertFalse(by_role['text']['available'])
        self.assertFalse(by_role['combined']['available'])
        self.assertEqual(result.json['primary_role'], 'metadata')

    def test_observed_rating_and_identifiers_do_not_enter_prediction(self):
        plain = self.client.post('/api/predict', json=self.payload)
        extras = self.client.post('/api/predict', json={**self.payload, 'rating': 10,
                                                       'high_rated': 1, 'show_id': 2})
        self.assertEqual(plain.status_code, 200)
        self.assertEqual(plain.json, extras.json)

    def test_invalid_runtime_gets_an_actionable_error(self):
        result = self.client.post('/api/predict', json={**self.payload, 'runtime_minutes': -5})
        self.assertEqual(result.status_code, 400)
        self.assertIn('Runtime minutes', result.json['error'])

    def test_examples_are_real_snapshot_rows(self):
        result = self.client.get('/api/examples')
        self.assertEqual(result.status_code, 200)
        self.assertEqual({row['show_id'] for row in result.json}, {2, 5, 185, 318})


if __name__ == '__main__':
    unittest.main()
