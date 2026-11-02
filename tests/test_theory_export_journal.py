import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from ml_lab.data import prepare_frame
from ml_lab.core import run_experiments
from ml_lab.experiments import ExperimentStore
from ml_lab.result_export import results_bundle
from ml_lab.model_theory import theory,architecture_dot,fitted_tree_dot
from ml_lab.model_registry import MODEL_REGISTRY


class TheoryAndExportTests(unittest.TestCase):
    def result(self):
        raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=100,freq='h'),'x':np.arange(100),'y':np.arange(100)*2})
        result=run_experiments(prepare_frame(raw,target='y'),['x'],'y','Регрессия',[{'name':'Decision Tree','params':{'max_depth':3}}],{'split':'time'})
        result['config']={'dataset':'test','seed':42}
        return result

    def test_every_available_model_has_theory_and_architecture(self):
        for name,spec in MODEL_REGISTRY.items():
            for task in spec['classes']:
                with self.subTest(name=name,task=task):
                    profile=theory(name,task)
                    self.assertTrue(profile['text']);self.assertTrue(profile['formula'])
                    self.assertIn('digraph G',architecture_dot(name,task))
        self.assertIn('Dense: 8',architecture_dot('MLP Neural Network','Регрессия',{'hidden_layer_sizes':[8,4]}))
        self.assertIn('слой 2',architecture_dot('LSTM','Регрессия',{'layers':2,'units':16}))

    def test_export_contains_actual_results_styled_charts_and_fitted_tree(self):
        result=self.result()
        before=result['predictions'].copy()
        data=results_bundle(result,{'train_test_timeline':{'title':'Мой временной график'}},include_model=False)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in ['metrics.json','test_predictions.csv','train_predictions.csv','validation_candidates_predictions.csv',
                'test_daily_metrics.csv','chart_styles.json','charts/train_test_timeline.html','charts/fitted_tree.dot','model_description.md']:
                self.assertIn(name,archive.namelist())
            self.assertNotIn('model.joblib',archive.namelist())
            exported=json.loads(archive.read('metrics.json'))
            self.assertEqual(exported['test'],result['metrics'])
            chart=json.loads(archive.read('charts/train_test_timeline.json'))
            self.assertEqual(chart['layout']['title']['text'],'Мой временной график')
            self.assertIn(b'plotly.js',archive.read('charts/train_test_timeline.html'))
        pd.testing.assert_frame_equal(before,result['predictions'])
        self.assertIn('x',fitted_tree_dot(result))

    def test_clear_journal_is_persistent_and_does_not_touch_other_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);marker=root/'dataset.csv';marker.write_text('preserve')
            store=ExperimentStore(root/'journal.sqlite3');result=self.result()
            store.save(result,result['config']);store.save(result,result['config'])
            self.assertEqual(store.clear(),2)
            self.assertEqual(ExperimentStore(store.path).list(),[])
            self.assertEqual(marker.read_text(),'preserve')
            self.assertEqual(store.clear(),0)

    def test_interface_theory_selection_and_clear_preserves_current_result(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as temporary:
            store=ExperimentStore(Path(temporary)/'journal.sqlite3')
            with patch('ml_lab.experiments.ExperimentStore',return_value=store):
                at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
                self.assertFalse(at.exception)
                self.assertIn('Описание моделей',[tab.label for tab in at.tabs])
                self.assertNotIn('Будущий прогноз',[tab.label for tab in at.tabs])
                at.multiselect(key='description_models_Регрессия').set_value(['Ridge','LSTM']).run(timeout=60)
                self.assertFalse(at.exception)
                next(w for w in at.selectbox if w.label=='Набор данных').select('MUF · временной ряд').run(timeout=60)
                at.selectbox(key='chosen_model').select('Ridge').run(timeout=60)
                next(w for w in at.button if w.label=='▶ Обучить и оценить').click().run(timeout=60)
                self.assertFalse(at.exception)
                metrics=dict(at.session_state.result['metrics'])
                self.assertEqual(len(store.list()),1)
                self.assertNotIn('Будущий прогноз',[tab.label for tab in at.tabs])
                at.checkbox(key='export_include_model').uncheck().run(timeout=60)
                at.button(key='prepare_results_export').click().run(timeout=60)
                self.assertFalse(at.exception)
                self.assertTrue(zipfile.is_zipfile(io.BytesIO(at.session_state.results_export['data'])))
                at.button(key='clear_experiment_journal').click().run(timeout=60)
                self.assertFalse(at.exception)
                self.assertEqual(store.list(),[])
                self.assertEqual(at.session_state.result['metrics'],metrics)
                self.assertEqual(at.session_state.history,[])
