import tempfile
import unittest
from pathlib import Path
import pandas as pd
from ml_lab.experiments import ExperimentStore, html_report


class JournalTests(unittest.TestCase):
    def result(self):
        return dict(name='<script>alert(1)</script>',task='Регрессия',target='y',features=['x'],
            metrics={'R²':float('nan'),'MAE':1.0},validation={'MAE':2.0},sizes={'train':30,'validation':10,'test':10},
            predictions=pd.DataFrame({'Факт':[1], 'Прогноз':[2]}),leaderboard=[{'Модель':'<unsafe>'}])

    def test_journal_survives_new_instance_and_preserves_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'runs.db'
            saved=ExperimentStore(path).save(self.result(),{'seed':42})
            loaded=ExperimentStore(path).list()[0]
            self.assertEqual(saved,loaded)
            self.assertIsNone(loaded['metrics']['R²'])
            self.assertEqual(loaded['config']['seed'],42)

    def test_report_escapes_dataset_text(self):
        report=html_report(self.result(),{'dataset':'<img src=x onerror=alert(1)>'}).decode()
        self.assertNotIn('<script>',report)
        self.assertNotIn('<img',report)
        self.assertIn('&lt;script&gt;',report)
        self.assertIn('Validation',report)
