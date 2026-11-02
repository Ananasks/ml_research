import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from ml_lab.data import prepare_frame,parse_datetime_values
from ml_lab.core import eligible_rows,run_experiments,split_data
from ml_lab.neural import RecurrentRegressor


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.raw=pd.DataFrame({'datetime':pd.date_range('2023-01-01',periods=240,freq='30min'),'y':np.arange(240,dtype=float)})

    def test_clock_only_is_never_a_date(self):
        text=pd.Series(['00:00','30:00','12:30:00','2023-01-01 00:00:00','02-01-2023 12:30:00'])
        for strict in [True,False]:
            time=parse_datetime_values(text,dayfirst=True,strict=strict)
            self.assertTrue(time.iloc[:3].isna().all())
            self.assertEqual(time.iloc[4],pd.Timestamp('2023-01-02 12:30:00'))

    def test_muf_strict_dates_and_grid(self):
        path=Path(__file__).resolve().parents[1]/'data'/'muf.csv'
        frame=prepare_frame(pd.read_csv(path),target='muf',drop_invalid_time=True,frequency='30min')
        self.assertEqual(frame.attrs['invalid_time_rows'],26)
        self.assertEqual(len(frame),17520)
        self.assertEqual(frame.index.max(),pd.Timestamp('2023-12-31 23:30:00'))
        self.assertEqual(frame.muf.isna().sum(),5116)

    def test_grid_keeps_unknown_targets_empty(self):
        raw=self.raw.drop(index=[4,5])
        frame=prepare_frame(raw,target='y',frequency='30min',lags=[1],rolling_windows=[3],difference=True)
        self.assertEqual(frame.attrs['grid_added_rows'],2)
        self.assertTrue(frame.y.iloc[4:6].isna().all())
        self.assertTrue(np.isnan(frame['y_lag_1'].iloc[6]))
        self.assertTrue(np.isnan(frame['y_rolling_mean_3'].iloc[8]))
        self.assertEqual(frame['y_rolling_mean_3'].iloc[9],7.)
        eligible=eligible_rows(frame,['y_lag_1','y_rolling_mean_3'],'y','Регрессия')
        self.assertNotIn(6,eligible.index)
        self.assertNotIn(8,eligible.index)

    def test_causal_features_do_not_change_with_future_targets(self):
        config=dict(target='y',frequency='30min',lags=[1,2,48],rolling_windows=[6,48],rolling_stats=['mean','std','min','max'],difference=True)
        frame=prepare_frame(self.raw,**config)
        changed=self.raw.copy();changed.loc[100:,'y']=999999
        other=prepare_frame(changed,**config)
        history=frame.attrs['history_columns']
        np.testing.assert_allclose(frame[history].iloc[:101].to_numpy(),other[history].iloc[:101].to_numpy(),equal_nan=True)
        self.assertEqual(frame['y_rolling_mean_6'].iloc[10],6.5)
        self.assertEqual(frame['y_past_difference'].iloc[10],1)
        self.assertEqual(frame['y_lag_48'].iloc[48],0)

    def test_grid_rejects_duplicates_and_off_grid_points(self):
        raw=pd.concat([self.raw,self.raw.iloc[:1]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'уникальные'):prepare_frame(raw,frequency='30min',target='y')
        frame=prepare_frame(raw,frequency='30min',duplicate_policy='last',target='y')
        self.assertEqual(frame.attrs['duplicates_removed'],1)
        raw=self.raw.copy();raw.loc[10,'datetime']+=pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError,'не совпадают'):prepare_frame(raw,frequency='30min',target='y')

    def test_history_requires_time_split(self):
        frame=prepare_frame(self.raw,target='y',rolling_windows=[6])
        with self.assertRaises(ValueError):split_data(frame,['y_rolling_mean_6'],'y','Регрессия',{'split':'random'})

    def test_baseline_uses_same_points_and_past_values(self):
        frame=prepare_frame(self.raw,target='y',frequency='30min',lags=[1],rolling_windows=[6])
        r=run_experiments(frame,['y_lag_1','y_rolling_mean_6'],'y','Регрессия',[{'name':'Ridge','params':{}}],{'split':'time'})
        self.assertNotIn('error',r)
        np.testing.assert_allclose(r['predictions']['Базовый прогноз'],r['predictions']['Факт']-1)
        self.assertEqual(r['baseline']['rows'],len(r['predictions']))
        self.assertAlmostEqual(r['baseline']['metrics']['MAE'],1.)

    def test_recurrent_windows_reset_at_gaps(self):
        model=RecurrentRegressor(window=3)
        values=np.array([[1.],[2.],[30.],[40.]])
        windows=model._windows(values,breaks=np.array([True,False,True,False])).numpy()
        np.testing.assert_array_equal(windows[2,:,0],[30,30,30])
        np.testing.assert_array_equal(windows[3,:,0],[30,30,40])

    def test_ui_preset_and_training(self):
        path=str(Path(__file__).resolve().parents[1]/'app.py')
        at=AppTest.from_file(path).run(timeout=60)
        next(w for w in at.selectbox if w.label=='Набор данных').select('MUF · временной ряд').run()
        next(w for w in at.button if w.label=='Настройки для MUF').click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(next(w for w in at.selectbox if w.label=='Модель').value,'Hist Gradient Boosting')
        self.assertTrue(next(w for w in at.checkbox if w.label=='Единая сетка времени').value)
        next(w for w in at.button if w.label=='▶ Обучить и оценить').click().run(timeout=90)
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        r=at.session_state.result
        self.assertTrue(r['settings']['rolling_windows'])
        self.assertIn('baseline',r)
        self.assertGreater(r['metrics']['R²'],.9)


if __name__=='__main__':unittest.main()
