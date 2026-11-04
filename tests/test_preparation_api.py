"""The data page must preview and export exactly what the training engine receives."""
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from api.main import (PreparationRequest, export_preparation, prepare_dataset,
                      preparation_params, preview_preparation)
from ml_lab.core import split_data
from ml_lab.data import prepare_frame


class PreparationAPITests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({'datetime': pd.date_range('2024-01-01', periods=240, freq='h'),
                                 'x': np.arange(240, dtype=float), 'y': np.arange(240, dtype=float) / 2})
        self.raw.loc[20, 'x'] = np.nan
        self.raw.loc[40, 'y'] = 1000
        self.options = dict(calendar=True, lags=[1, 3], rolling_windows=[4],
                            rolling_stats=['mean', 'trend'], ewm_spans=[3], difference_lags=[2],
                            horizon=2, quality_flags=True, drop_invalid_time=True,
                            repair=dict(columns=['x', 'y'], window=48, min_history=8, threshold=3,
                                        fill_method='linear', anomaly_method='iqr', anomaly_action='clip'))

    def preview(self, request):
        with patch('api.main.get_dataset', return_value=self.raw):
            return preview_preparation('sample.csv', request)

    def test_preview_matches_shared_preparation_and_split(self):
        expected = prepare_frame(self.raw, target='y', **self.options)
        features = ['x', 'y_lag_3', 'calendar_hour', 'y_was_anomaly']
        request = PreparationRequest(target='y', features=features, preparation=self.options)
        actual = self.preview(request)
        self.assertEqual(actual['features'], [c for c in expected if c != 'y'])
        self.assertEqual(actual['repairRecords'], expected.attrs['repair_records'])
        self.assertIsNone(actual['splitError'])
        _, _, indices = split_data(expected, features, 'y', 'Регрессия', {})
        self.assertEqual([actual['split'][k] for k in ('train', 'validation', 'test')], list(map(len, indices)))
        self.assertEqual(actual['split']['unused'], actual['split']['outside']+actual['split']['notEligible'])
        series = actual['repairSeries']['y']
        self.assertEqual(series[40]['before'], 1000)
        self.assertTrue(series[40]['anomaly'])
        self.assertEqual(series[40]['after'], expected.attrs['repair_values']['y'].iloc[40])
        json.dumps(jsonable_encoder(actual), allow_nan=False)

    def test_export_contains_all_prepared_rows_and_real_datetime(self):
        request = PreparationRequest(target='y', preparation=self.options)
        with patch('api.main.get_dataset', return_value=self.raw):
            response = export_preparation('sample.csv', request)
        actual = pd.read_csv(io.BytesIO(response.body), parse_dates=['datetime'])
        expected = prepare_frame(self.raw, target='y', **self.options).reset_index()
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_row_counts_for_calendar_period_match_training(self):
        prepared, _ = prepare_dataset(self.raw, 'y', 'Регрессия', self.options)
        features = ['x', 'y_lag_3']
        options = dict(size_mode='rows', train_rows=80, validation_rows=30, test_rows=1,
                       forecast_start='2024-01-08', forecast_end='2024-01-08')
        effective = preparation_params(prepared, features, 'y', 'Регрессия', options)
        self.assertEqual(effective['test_rows'], 24)
        actual = self.preview(PreparationRequest(target='y', features=features, preparation=self.options, params=options))
        self.assertIsNone(actual['splitError'])
        self.assertEqual(actual['split']['test'], 24)

    def test_invalid_split_does_not_hide_prepared_data(self):
        actual = self.preview(PreparationRequest(target='y', features=['x'], preparation=self.options,
                              params=dict(size_mode='rows', train_rows=1000, validation_rows=30, test_rows=10)))
        self.assertTrue(actual['rows'])
        self.assertIsNone(actual['split'])
        self.assertIn('Доступно', actual['splitError'])

    def test_stale_column_and_classification_target_are_rejected(self):
        for options, task in [(dict(repair=dict(columns=['CS'])), 'Регрессия'),
                              (dict(repair=dict(columns=['y'])), 'Классификация')]:
            with self.subTest(task=task), self.assertRaises(HTTPException) as error:
                self.preview(PreparationRequest(target='y', task=task, preparation=options))
            self.assertEqual(error.exception.status_code, 422)

    def test_generated_index_switch_and_time_grid_diagnostics(self):
        raw = self.raw.drop(columns=['datetime'])
        prepared, _ = prepare_dataset(raw, 'y', 'Регрессия', dict(generate_index=False, start_datetime='2024-01-01', index_frequency='1h'))
        self.assertNotIsInstance(prepared.index, pd.DatetimeIndex)
        prepared, _ = prepare_dataset(raw, 'y', 'Регрессия', dict(generate_index=True, start_datetime='2024-01-01', index_frequency='1h'))
        self.assertIsInstance(prepared.index, pd.DatetimeIndex)
        self.raw = self.raw.drop(index=[5]).reset_index(drop=True)
        self.raw.loc[len(self.raw)] = ['invalid', 0, 0]
        actual = self.preview(PreparationRequest(target='y', preparation=dict(frequency='1h', drop_invalid_time=True)))
        self.assertEqual(actual['diagnostics']['invalid_time_rows'], 1)
        self.assertEqual(actual['diagnostics']['grid_added_rows'], 1)


if __name__ == '__main__':
    unittest.main()
