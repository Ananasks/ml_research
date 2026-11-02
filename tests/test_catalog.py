import io
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from ml_lab.adapters import EncodedClassifier
from ml_lab.model_registry import available_models,MODEL_REGISTRY,OPTIONAL_MODELS,documentation,catalog_rows
from ml_lab.core import run_experiments,make_pipeline
from ml_lab.data import prepare_frame


class CatalogueTests(unittest.TestCase):
    def test_filter_respects_task_category_query_and_sequence_constraints(self):
        neural=available_models('Регрессия','time','Нейронные сети')
        self.assertIn('LSTM',neural)
        self.assertNotIn('LSTM',available_models('Регрессия','random','Нейронные сети'))
        self.assertNotIn('LSTM',available_models('Классификация','time'))
        self.assertEqual(available_models('Регрессия','time','Деревья и ансамбли','Extra Tree'),['Extra Trees','Extra Tree'])
        self.assertGreater(len(available_models('Регрессия')),27)
        self.assertTrue(all(row['Документация'].startswith('https://') for row in catalog_rows('Регрессия')))

    def test_encoded_classifier_preserves_string_labels_probability_order_and_export(self):
        X=np.arange(60).reshape(-1,1);y=np.tile(['low','high','mid'],20)
        model=EncodedClassifier(RandomForestClassifier(n_estimators=5,random_state=2))
        fitted=clone(model).fit(X,y)
        probability=fitted.predict_proba(X)
        np.testing.assert_array_equal(fitted.predict(X),fitted.classes_[probability.argmax(axis=1)])
        stream=io.BytesIO();joblib.dump(fitted,stream);stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict(X),fitted.predict(X))

    def test_expensive_kernel_model_checks_training_size(self):
        with self.assertRaisesRegex(ValueError,'2000'):
            make_pipeline(pd.DataFrame({'x':np.arange(2001)}),'Регрессия','Gaussian Process',{})

    def test_installed_extensions_fit_both_tasks_with_original_class_labels(self):
        raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=120,freq='h'),'x':np.tile([-2.,-1.,1.,2.],30),'y':np.tile([-4.,-2.,2.,4.],30)})
        tested=0
        for name in OPTIONAL_MODELS:
            if name not in MODEL_REGISTRY: continue
            for task in ['Регрессия','Классификация']:
                with self.subTest(model=name,task=task):
                    source=raw.copy()
                    if task=='Классификация': source['y']=np.where(source.x>0,'positive','negative')
                    frame=prepare_frame(source,target='y')
                    result=run_experiments(frame,['x'],'y',task,[{'name':name,'params':{'n_estimators':10,'iterations':10}}],{'split':'time'})
                    self.assertNotIn('error',result,str(result))
                    self.assertTrue(all(np.isfinite(v) for v in result['metrics'].values()))
                    if task=='Классификация':
                        self.assertEqual(set(result['predictions']['Прогноз']),{'positive','negative'})
                        np.testing.assert_allclose(result['test_probabilities'].sum(axis=1),1.,rtol=1e-5)
                    self.assertIn('https://',documentation(name,task))
                    tested+=1
        if not tested: self.skipTest('Optional model packages are not installed')

    def test_category_selection_and_search_in_interface(self):
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
        at.selectbox(key='model_category_Регрессия').select('Деревья и ансамбли').run(timeout=60)
        self.assertFalse(at.exception)
        self.assertTrue(all(MODEL_REGISTRY[name]['group']=='Деревья и ансамбли' for name in at.selectbox(key='chosen_model').options))
        at.text_input(key='model_search_Регрессия').set_value('Extra Tree').run(timeout=60)
        self.assertEqual(set(at.selectbox(key='chosen_model').options),{'Extra Tree','Extra Trees'})
        at.text_input(key='model_search_Регрессия').set_value('').run(timeout=60)
        at.selectbox(key='model_category_Регрессия').select('Нейронные сети').run(timeout=60)
        self.assertEqual(at.selectbox(key='chosen_model').options,['MLP Neural Network'])
