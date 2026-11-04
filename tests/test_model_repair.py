"""Learned recovery must be useful without using future or fabricated labels."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from ml_lab.data import prepare_frame
from ml_lab.repair import repair_signals
from ml_lab.model_repair import LearnedRepair


def config(model='ridge', **kwargs):
    learned=dict(model=model,lags=3,min_samples=32,train_window=96,refit_interval=16,max_fits=3,
                 iterations=30,trees=10,epochs=2,units=8,sequence=3,fallback='none')
    learned.update(kwargs)
    return dict(columns=['y'],fill_method='model',anomaly_method='none',model_config=learned)


class ModelRepairTests(unittest.TestCase):
    def raw(self, n=180):
        return pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=n,freq='h'),
                             'x':np.arange(n,dtype=float),'y':5+np.arange(n)*.1+np.sin(np.arange(n)/6)})

    def test_each_model_recovers_a_gap_with_a_real_fit(self):
        for model in ['ridge','random_forest','extra_trees','hist_gradient_boosting','mlp','lstm','gru']:
            with self.subTest(model=model):
                raw=self.raw(); raw.loc[100:103,'y']=np.nan
                frame=prepare_frame(raw,target='y',repair=config(model))
                record=frame.attrs['repair_records'][0]
                self.assertTrue(np.isfinite(frame.y.iloc[100:104]).all())
                self.assertEqual(record['Восстановлено моделью'],4)
                self.assertEqual(record['Обучений модели'],1)
                self.assertEqual(record['Резервным способом'],0)
                self.assertTrue(frame.attrs['evaluation_target'].iloc[100:104].isna().all())

    def test_future_values_cannot_change_past_repairs(self):
        for model in ['ridge','random_forest','mlp','lstm','gru']:
            with self.subTest(model=model):
                raw=self.raw(); raw.loc[80:83,'y']=np.nan;raw.loc[135:138,'y']=np.nan
                changed=raw.copy();changed.loc[110:,'y']=999999
                first=prepare_frame(raw,target='y',repair=config(model))
                second=prepare_frame(changed,target='y',repair=config(model))
                np.testing.assert_allclose(first.y.iloc[:110],second.y.iloc[:110],equal_nan=True)

    def test_training_labels_are_only_past_observed_non_anomalous_values(self):
        raw=self.raw();raw.loc[80:84,'y']=np.nan;raw.loc[100,'y']=999999;raw.loc[130:132,'y']=np.nan
        captures=[]
        original_build=LearnedRepair.build
        def build(learner):
            captures.append((learner.positions.copy(),learner.y.copy()))
            return original_build(learner)
        settings=config(refit_interval=1);settings.update(anomaly_action='replace',bounds={'y':{'min':0,'max':100}})
        with patch.object(LearnedRepair,'build',build):
            frame=prepare_frame(raw,target='y',repair=settings)
        self.assertTrue(captures)
        for positions,labels in captures:
            self.assertTrue(set(positions).isdisjoint({80,81,82,83,84,100,130,131,132}))
            np.testing.assert_allclose(labels,raw.y.iloc[positions].to_numpy())
        self.assertEqual(frame.attrs['repair_records'][0]['Обучений модели'],3)

    def test_initial_gaps_max_gap_bounds_and_explicit_fallback(self):
        raw=self.raw();raw.loc[:2,'y']=np.nan;raw.loc[10:12,'y']=np.nan;raw.loc[100:105,'y']=np.nan
        settings=config(fallback='forward');settings.update(max_gap=2,bounds={'y':{'max':15}})
        frame=prepare_frame(raw,target='y',repair=settings)
        self.assertTrue(frame.y.iloc[:3].isna().all())
        self.assertTrue(frame.y.iloc[10:12].notna().all())
        self.assertTrue(pd.isna(frame.y.iloc[12]))
        self.assertTrue(frame.y.iloc[102:106].isna().all())
        self.assertTrue((frame.y.dropna()<=15).all())
        self.assertGreaterEqual(frame.attrs['repair_records'][0]['Резервным способом'],2)

    def test_insufficient_history_can_remain_missing_and_invalid_options_fail(self):
        raw=self.raw(20);raw.loc[10,'y']=np.nan
        frame=prepare_frame(raw,target='y',repair=config())
        self.assertTrue(pd.isna(frame.y.iloc[10]))
        for options in [dict(model='unknown'),dict(lags=0),dict(train_window=16),dict(min_samples=16,sequence=24,model='lstm'),dict(fallback='future')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                prepare_frame(raw,target='y',repair=config(**options))

    def test_no_gap_does_not_fit_a_model(self):
        with patch.object(LearnedRepair,'build',side_effect=AssertionError('Unnecessary fit')):
            frame=prepare_frame(self.raw(),target='y',repair=config())
        self.assertEqual(frame.attrs['repair_records'][0]['Обучений модели'],0)

    def test_fit_failure_is_reported_and_uses_configured_reserve(self):
        raw=self.raw();raw.loc[80:82,'y']=np.nan
        with patch.object(LearnedRepair,'build',side_effect=ValueError('fit failed')):
            frame=prepare_frame(raw,target='y',repair=config(fallback='forward'))
        record=frame.attrs['repair_records'][0]
        self.assertEqual(record['Обучений модели'],0)
        self.assertEqual(record['Резервным способом'],3)
        self.assertIn('fit failed',record['Причина резерва'])
        np.testing.assert_allclose(frame.y.iloc[80:83],raw.y.iloc[79])

    def test_api_reuses_preparation_but_invalidates_changed_source(self):
        from api.main import prepare_dataset, _PREPARED_MODELS
        _PREPARED_MODELS.clear()
        raw=self.raw();raw.loc[80:82,'y']=np.nan
        original=repair_signals
        with patch('ml_lab.data.repair_signals',wraps=original) as repair:
            first,_=prepare_dataset(raw,'y','Регрессия',{'repair':config()})
            first.iloc[0,0]=-999
            again,_=prepare_dataset(raw,'y','Регрессия',{'repair':config()})
            self.assertEqual(repair.call_count,1)
            self.assertNotEqual(again.iloc[0,0],-999)
            changed=raw.copy();changed.loc[20,'y']+=1
            prepare_dataset(changed,'y','Регрессия',{'repair':config()})
            self.assertEqual(repair.call_count,2)
        _PREPARED_MODELS.clear()
