import unittest
import numpy as np
import pandas as pd
from ml_lab.data import prepare_frame
from ml_lab.core import run_experiments, evaluate, split_data
from ml_lab.forecast import forecast_dates
from ml_lab.explain import permutation_scores


def settings(columns=('x','y'),**kwargs):
    return dict(columns=list(columns),fill_method='linear',anomaly_method='iqr',anomaly_action='clip',window=12,min_history=4,threshold=3.,**kwargs)


class RepairTests(unittest.TestCase):
    def raw(self, count=180):
        return pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=count,freq='h'),
            'x':np.arange(count,dtype=float),'y':2*np.arange(count,dtype=float)+1})

    def test_extrapolation_uses_actual_time_and_preserves_rows(self):
        raw=pd.DataFrame({'datetime':pd.to_datetime(['2024-01-01 00:00','2024-01-01 01:00','2024-01-01 03:00','2024-01-01 06:00']),
            'x':[0.,2.,np.nan,np.nan],'y':[1.,3.,np.nan,np.nan]})
        frame=prepare_frame(raw,target='y',repair=settings())
        np.testing.assert_allclose(frame.x,[0,2,6,12])
        np.testing.assert_allclose(frame.y,[1,3,7,13])
        self.assertEqual(len(frame),len(raw))
        self.assertTrue(frame.attrs['evaluation_target'].iloc[-2:].isna().all())
        self.assertIsInstance(frame.index,pd.DatetimeIndex)

    def test_detection_and_history_do_not_depend_on_future(self):
        raw=self.raw();raw.loc[20,'x']=99999;raw.loc[21:23,'y']=np.nan
        config=dict(target='y',repair=settings(),quality_flags=True,feature_columns=['x','y'],lags=[1,3],
            rolling_windows=[4],rolling_stats=['median','iqr','trend'],ewm_spans=[4],difference_lags=[2])
        frame=prepare_frame(raw,**config)
        changed=raw.copy();changed.loc[80:,['x','y']]=-999999
        other=prepare_frame(changed,**config)
        columns=['x','y']+frame.attrs['history_columns']
        np.testing.assert_allclose(frame[columns].iloc[:80],other[columns].iloc[:80],equal_nan=True)
        self.assertLess(frame.x.iloc[20],100.)
        self.assertTrue(frame.attrs['repair_masks']['x']['anomaly'][20])
        self.assertEqual(frame['y_was_missing'].iloc[21],0)
        self.assertEqual(frame['y_was_missing'].iloc[22],1)

    def test_all_detectors_restore_a_spike_without_removing_the_row(self):
        for method in ['iqr','mad','zscore']:
            with self.subTest(method=method):
                raw=self.raw();raw.x=5.;raw.loc[20,'x']=1000.
                config=settings(columns=['x']);config.update(anomaly_method=method,anomaly_action='replace')
                frame=prepare_frame(raw,target='y',repair=config)
                self.assertEqual(frame.x.iloc[20],5.)
                self.assertEqual(len(frame),len(raw))
                self.assertTrue(frame.attrs['repair_masks']['x']['corrected'][20])

    def test_initial_missing_and_long_gaps_are_not_backfilled(self):
        raw=self.raw();raw.loc[:1,'x']=np.nan;raw.loc[4:8,'x']=np.nan
        frame=prepare_frame(raw,target='y',repair=settings(columns=['x'],max_gap=2))
        self.assertTrue(frame.x.iloc[:2].isna().all())
        self.assertTrue(frame.x.iloc[4:6].notna().all())
        self.assertTrue(frame.x.iloc[6:9].isna().all())
        self.assertEqual(frame.x.iloc[9],9.)

    def test_physical_bounds_and_multiple_source_features(self):
        raw=self.raw();raw.loc[30,'x']=-300
        frame=prepare_frame(raw,target='y',repair=settings(columns=['x'],bounds={'x':{'min':0,'max':100}}),
            feature_columns=['x'],lags=[2],rolling_windows=[4],rolling_stats=['median','iqr','trend'],ewm_spans=[4],difference_lags=[2])
        self.assertGreaterEqual(frame.x.iloc[30],0.)
        self.assertLess(frame.x.iloc[30],100.)
        self.assertTrue((frame.x<=100).all())
        self.assertEqual(frame.x_lag_2.iloc[10],8.)
        self.assertEqual(frame.x_rolling_median_4.iloc[10],7.5)
        self.assertEqual(frame.x_rolling_trend_4.iloc[10],1.)
        self.assertEqual(frame.x_past_difference_2.iloc[10],2.)
        self.assertLess(frame.x_ewm_4.iloc[10],10.)
        with self.assertRaisesRegex(ValueError,'по времени'):
            split_data(frame,['x_lag_2'],'y','Регрессия',{'split':'random'})

    def test_synthetic_targets_do_not_enter_validation_or_test_metrics(self):
        raw=self.raw();raw.loc[[15,115,155],'y']=np.nan
        config=dict(target='y',repair=settings(),feature_columns=['x'])
        frame=prepare_frame(raw,**config)
        result=run_experiments(frame,['x'],'y','Регрессия',[{'name':'Ridge','params':{}}],{'split':'time'})
        self.assertNotIn('error',result)
        self.assertEqual(result['evaluation_sizes'],{'validation':35,'test':35,'restored_train':1})
        self.assertEqual(result['predictions']['Факт'].isna().sum(),1)
        self.assertFalse(result['predictions'].loc[pd.Timestamp('2024-01-07 11:00'),'Факт измерен'])
        expected=evaluate(result['predictions']['Факт'],result['predictions']['Прогноз'],'Регрессия')
        self.assertEqual(expected,result['metrics'])
        self.assertTrue(permutation_scores(result)['Важность'].notna().all())

    def test_future_known_x_reuses_history_for_lags_and_repair(self):
        raw=self.raw()
        config=dict(target='y',time_columns=['datetime'],calendar=True,repair=settings(),feature_columns=['x'],
            lags=[1,2],rolling_windows=[3],ewm_spans=[3],difference_lags=[1],quality_flags=True,horizon=0)
        frame=prepare_frame(raw,**config)
        features=['x_lag_1','x_rolling_mean_3','x_ewm_3','x_past_difference','calendar_hour']
        result=run_experiments(frame,features,'y','Регрессия',[{'name':'Ridge','params':{}}],{'split':'time'})
        self.assertNotIn('error',result)
        result['settings']=config
        dates=pd.date_range(raw.datetime.iloc[-1]+pd.Timedelta(hours=1),periods=3,freq='h')
        inputs=pd.DataFrame({'datetime':dates,'x':[180.,np.nan,182.]})
        future=forecast_dates(result,raw,dates[0],dates[-1],'1h',inputs)
        self.assertEqual(len(future),3)
        self.assertTrue(np.isfinite(future['Прогноз']).all())
        result['features']=['y_was_missing']
        with self.assertRaisesRegex(ValueError,'истории цели'):
            forecast_dates(result,raw,dates[0],dates[-1],'1h',inputs)

class RepairInterfaceTests(unittest.TestCase):
    def test_uploaded_repair_and_extra_history_features_train(self):
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        from ml_lab.datasets import dataset_record
        raw=RepairTests().raw();raw.loc[[15,115,155],'y']=np.nan;raw.loc[20,'x']=99999
        record=dataset_record(raw.to_csv(index=False).encode(),'repair-case.csv')
        at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
        at.session_state.datasets_session={record['id']:record}
        at.radio(key='datasets_source').set_value('Загруженные и сохранённые').run(timeout=60)
        at.multiselect(key='datasets_selected').set_value([record['id']]).run(timeout=60)
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        next(w for w in at.multiselect if w.label=='Колонки для признаков истории').set_value(['x']).run()
        next(w for w in at.multiselect if w.label=='Экспоненциальное среднее · span').set_value([3]).run()
        next(w for w in at.multiselect if w.label=='Разности прошлых значений · лаг').set_value([1]).run()
        next(w for w in at.selectbox if w.label=='Модель').select('Ridge').run()
        next(w for w in at.button if w.label=='▶ Обучить и оценить').click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        result=at.session_state.result
        self.assertIn('x_ewm_3',result['features'])
        self.assertIn('x_past_difference',result['features'])
        self.assertGreater(result['evaluation_sizes']['restored_train'],0)
        self.assertLess(result['evaluation_sizes']['test'],result['sizes']['test'])
