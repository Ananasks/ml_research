import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from ml_lab.data import prepare_frame
from ml_lab.anomaly_detection import MODEL_DETECTORS, detect_model_anomalies


def settings(method, **options):
    detector=dict(min_samples=32,train_window=96,refit_interval=16,max_fits=12,trees=20,contamination=.05,seed=42)
    detector.update(options)
    return dict(columns=['y'],fill_method='linear',anomaly_method=method,anomaly_action='replace',anomaly_config=detector)


class AnomalyModelTests(unittest.TestCase):
    def raw(self, count=240):
        return pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=count,freq='h'),
                             'y':10+np.sin(np.arange(count)/3)})

    def test_each_detector_identifies_and_replaces_a_spike(self):
        for method in MODEL_DETECTORS:
            with self.subTest(method=method):
                raw=self.raw();raw.loc[100,'y']=1000
                frame=prepare_frame(raw,target='y',repair=settings(method))
                masks=frame.attrs['repair_masks']['y']
                self.assertTrue(masks['anomaly'][100])
                self.assertTrue(masks['corrected'][100])
                self.assertLess(frame.y.iloc[100],100)
                self.assertGreater(masks['anomaly_score'][100],0)
                self.assertGreater(frame.attrs['repair_records'][0]['Обучений детектора'],0)
                self.assertEqual(frame.attrs['evaluation_target'].iloc[100],1000)

    def test_future_measurements_cannot_change_past_flags_or_repairs(self):
        for method in MODEL_DETECTORS:
            with self.subTest(method=method):
                raw=self.raw();raw.loc[100,'y']=1000;raw.loc[101,'y']=np.nan
                changed=raw.copy();changed.loc[110:,'y']=1000000
                a=prepare_frame(raw,target='y',repair=settings(method))
                b=prepare_frame(changed,target='y',repair=settings(method))
                np.testing.assert_array_equal(a.attrs['repair_masks']['y']['anomaly'][:110],b.attrs['repair_masks']['y']['anomaly'][:110])
                np.testing.assert_allclose(a.y.iloc[:110],b.y.iloc[:110],equal_nan=True)
                np.testing.assert_allclose(a.attrs['repair_masks']['y']['anomaly_score'][:110],b.attrs['repair_masks']['y']['anomaly_score'][:110],equal_nan=True)

    def test_fit_uses_prior_measurements_without_extreme_spikes(self):
        raw=self.raw();raw.loc[40,'y']=1000;raw.loc[50,'y']=np.nan
        captured=[]
        class Detector:
            def fit(self, x): captured.append(x.copy());return self
            def decision_function(self, x): return np.where(np.abs(x[:,0])>100,-1.,1.)
        with patch('ml_lab.anomaly_detection.build_detector',return_value=Detector()):
            prepare_frame(raw,target='y',repair=settings('isolation_forest'))
        np.testing.assert_allclose(captured[0].ravel(),raw.y.iloc[:32])
        for values in captured:
            self.assertTrue(np.isfinite(values).all())
            self.assertFalse((values==1000).any())
            self.assertLessEqual(len(values),96)
        # A synthetic interpolation at row 50 cannot be part of detector fitting.
        original=set(raw.y.dropna())
        self.assertTrue(all(float(value) in original for values in captured for value in values.ravel()))

    def test_constant_history_physical_limits_and_missing_values(self):
        for method in MODEL_DETECTORS:
            with self.subTest(method=method):
                raw=self.raw();raw.y=10.;raw.loc[80,'y']=1000;raw.loc[90,'y']=np.nan;raw.loc[5,'y']=-100
                config=settings(method);config['bounds']={'y':{'min':0,'max':20}}
                frame=prepare_frame(raw,target='y',repair=config)
                self.assertTrue(frame.attrs['repair_masks']['y']['anomaly'][80])
                self.assertEqual(frame.y.iloc[80],10)
                self.assertEqual(frame.y.iloc[90],10)
                self.assertTrue((frame.y.dropna().between(0,20)).all())
                self.assertIn('постоянного',frame.attrs['repair_records'][0]['Диагностика детектора'])

    def test_detector_budget_and_failures_are_reported(self):
        raw=self.raw();raw.loc[100,'y']=1000
        frame=prepare_frame(raw,target='y',repair=settings('isolation_forest',max_fits=2))
        record=frame.attrs['repair_records'][0]
        self.assertEqual(record['Обучений детектора'],2)
        self.assertTrue(record['Лимит обновлений детектора достигнут'])
        with patch('ml_lab.anomaly_detection.build_detector',side_effect=ValueError('bad fit')):
            frame=prepare_frame(raw,target='y',repair=settings('isolation_forest',max_fits=1))
        self.assertIn('Резерв IQR',frame.attrs['repair_records'][0]['Диагностика детектора'])
        self.assertTrue(frame.attrs['repair_masks']['y']['anomaly'][100])

    def test_detection_combines_with_learned_reconstruction(self):
        raw=self.raw();raw.loc[100,'y']=1000
        config=settings('isolation_forest');config.update(fill_method='model',model_config=dict(model='ridge',min_samples=16,lags=3,max_fits=2))
        frame=prepare_frame(raw,target='y',repair=config)
        record=frame.attrs['repair_records'][0]
        self.assertEqual(record['Детектор аномалий'],'Isolation Forest')
        self.assertGreater(record['Восстановлено моделью'],0)
        self.assertLess(frame.y.iloc[100],100)

    def test_rolling_support_does_not_collapse_after_repeated_tail_flags(self):
        raw=self.raw(3000)
        frame=prepare_frame(raw,target='y',repair=settings('isolation_forest',refit_interval=64,max_fits=64))
        count=int(frame.attrs['repair_masks']['y']['anomaly'].sum())
        self.assertLess(count, len(frame)*.2)

    def test_invalid_detector_settings_are_rejected(self):
        for options in [dict(contamination=0),dict(contamination=.6),dict(min_samples=64,train_window=32),dict(refit_interval=0)]:
            with self.subTest(options=options),self.assertRaises(ValueError):
                prepare_frame(self.raw(),target='y',repair=settings('isolation_forest',**options))

    def test_future_scoring_failure_does_not_change_earlier_decisions(self):
        raw=self.raw(100).y.to_numpy();changed=raw.copy();changed[60]=1000000
        class Detector:
            def fit(self,x): return self
            def decision_function(self,x):
                if (x>100).any():raise ValueError('bad query')
                return np.ones(len(x))
        with patch('ml_lab.anomaly_detection.build_detector',return_value=Detector()):
            a=detect_model_anomalies(raw,np.zeros(100,bool),'isolation_forest',dict(min_samples=32,refit_interval=64,max_fits=1))
            b=detect_model_anomalies(changed,np.zeros(100,bool),'isolation_forest',dict(min_samples=32,refit_interval=64,max_fits=1))
        np.testing.assert_allclose(a[1][:60],b[1][:60],equal_nan=True)
