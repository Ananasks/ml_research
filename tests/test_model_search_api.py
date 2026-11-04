"""Previewed configurations must be the configurations evaluated by the model page."""
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

from api.main import CandidateRequest, TrainRequest, RUNS, RUN_CONFIGS, candidates, train


class ModelSearchAPITests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({'x': range(120), 'y': [3 * x + 10 for x in range(120)]})
        self.run_ids = set(RUNS)

    def tearDown(self):
        for key in set(RUNS) - self.run_ids:
            RUNS.pop(key, None)
            RUN_CONFIGS.pop(key, None)

    def test_preview_uses_unique_reproducible_choices(self):
        request = CandidateRequest(task='Регрессия', models=['Linear Regression'], count=5,
                                   params={'seed': 42}, spaces={'Linear Regression': {
                                       'fit_intercept': {'type': 'choice', 'choices': [True, False]}}})
        first = candidates(request)
        self.assertEqual(first, candidates(request))
        self.assertEqual(len(first), 2)
        self.assertEqual({item['params']['fit_intercept'] for item in first}, {True, False})

    def test_training_evaluates_exact_preview_list(self):
        shown = [{'name': 'Linear Regression', 'params': {'fit_intercept': False}},
                 {'name': 'Linear Regression', 'params': {'fit_intercept': True}}]
        request = TrainRequest(dataset='example.csv', target='y', features=['x'],
                               model='Linear Regression', strategy='range', count=5,
                               preparation={'calendar': False}, candidates=shown)
        with patch('api.main.get_dataset', return_value=self.raw), patch('api.main.STORE.save', return_value='test'):
            result = train(request)
        self.assertEqual(len(result['leaderboard']), 2)
        self.assertEqual([row['Параметры']['fit_intercept'] for row in result['leaderboard']], [False, True])
        self.assertEqual(result['config']['candidates'], shown)
        self.assertTrue(all(row['Статус'] == 'Готово' for row in result['leaderboard']))

    def test_foreign_models_and_unknown_parameters_are_rejected(self):
        for candidate in [{'name': 'Ridge', 'params': {}},
                          {'name': 'Linear Regression', 'params': {'seed': 999}}]:
            request = TrainRequest(dataset='example.csv', target='y', features=['x'],
                                   model='Linear Regression', strategy='range', candidates=[candidate])
            with self.subTest(candidate=candidate), patch('api.main.get_dataset', return_value=self.raw):
                with self.assertRaises(HTTPException) as error:
                    train(request)
                self.assertEqual(error.exception.status_code, 422)


if __name__ == '__main__':
    unittest.main()
