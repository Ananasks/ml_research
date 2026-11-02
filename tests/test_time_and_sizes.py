import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ml_lab.data import prepare_frame
from ml_lab.core import split_data, run_experiments
from streamlit.testing.v1 import AppTest


class TimeAndSizeTests(unittest.TestCase):
    def setUp(self):
        self.raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=240,freq='h'),'x':np.arange(240),'y':np.sin(np.arange(240))})
        self.p={'size_mode':'rows','train_rows':100,'validation_rows':30,'test_rows':20,'split':'time'}

    def test_automatic_index_and_full_predictors(self):
        frame=prepare_frame(self.raw,target='y')
        self.assertIsInstance(frame.index,pd.DatetimeIndex)
        self.assertEqual(frame.index.name,'datetime')
        for feature in ['hour','minute','dayofweek','quarter','month','dayofyear','day','weekofyear']:
            self.assertIn('calendar_'+feature,frame)
        self.assertEqual(frame.iloc[0].calendar_quarter,1)
        self.assertEqual(frame.iloc[0].calendar_dayofweek,0)
        self.assertEqual(frame.iloc[0].calendar_weekofyear,1)

    def test_datetime_takes_precedence_over_date_time(self):
        raw=self.raw.assign(date='BAD',time='BAD')
        frame=prepare_frame(raw,['date','time'],target='y')
        self.assertEqual(frame.index[0],pd.Timestamp('2024-01-01'))
        raw=self.raw.drop(columns='datetime').assign(date=self.raw.datetime.dt.strftime('%d-%m-%Y'),time=self.raw.datetime.dt.strftime('%H:%M:%S'))
        frame=prepare_frame(raw,dayfirst=True,target='y')
        self.assertEqual(frame.index[1],pd.Timestamp('2024-01-01 01:00:00'))

    def test_exact_counts_first_and_last(self):
        frame=prepare_frame(self.raw,target='y')
        for window,first in [('first',0),('last',90)]:
            X,y,parts=split_data(frame,['x'],'y','Регрессия',self.p|{'row_window':window})
            self.assertEqual(list(map(len,parts)),[100,30,20])
            self.assertEqual(X.iloc[parts[0]].index[0],self.raw.datetime.iloc[first])
        with self.assertRaisesRegex(ValueError,'запрошено'):
            split_data(frame,['x'],'y','Регрессия',self.p|{'train_rows':230})

    def test_counts_horizon_gap(self):
        frame=prepare_frame(self.raw,target='y',horizon=3)
        X,y,(tr,va,te)=split_data(frame,['x'],'y','Регрессия',self.p)
        self.assertEqual([len(tr),len(va),len(te)],[100,30,20])
        self.assertLess(X.iloc[tr].index[-1]+pd.Timedelta(hours=3),X.iloc[va].index[0])
        self.assertLess(X.iloc[va].index[-1]+pd.Timedelta(hours=3),X.iloc[te].index[0])

    def test_exact_counts_random_classification(self):
        raw=self.raw.copy();raw['y']=np.where(raw.x%2,'a','b')
        frame=prepare_frame(raw,target='y')
        X,y,parts=split_data(frame,['x'],'y','Классификация',self.p|{'split':'random'})
        self.assertEqual(list(map(len,parts)),[100,30,20])
        self.assertTrue(all(y.iloc[part].nunique()==2 for part in parts))
        self.assertFalse(set(parts[0])&set(parts[1]))

    def test_counts_before_forecast_period(self):
        frame=prepare_frame(self.raw,target='y')
        p=self.p|{'forecast_start':'2024-01-08','forecast_end':'2024-01-08 23:00:00','test_rows':24}
        r=run_experiments(frame,['x'],'y','Регрессия',[{'name':'Ridge','params':p}],p)
        self.assertNotIn('error',r)
        self.assertEqual(r['sizes'],{'train':100,'validation':30,'test':24})
        self.assertTrue((r['predictions']['Время'].dt.day==8).all())
        with self.assertRaisesRegex(ValueError,'тестовых строк'):
            split_data(frame,['x'],'y','Регрессия',p|{'test_rows':20})

    def test_duplicate_time_boundary_rejected_for_exact_counts(self):
        raw=self.raw.copy();raw.loc[100,'datetime']=raw.loc[99,'datetime']
        frame=prepare_frame(raw,target='y')
        with self.assertRaisesRegex(ValueError,'одинаковые временные'):
            split_data(frame,['x'],'y','Регрессия',self.p)

    def test_ui_auto_time_and_editable_counts(self):
        at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
        self.assertFalse(any(w.label=='Колонки даты / времени' for w in at.multiselect))
        self.assertFalse(any(w.label=='Указать нестандартные колонки времени' for w in at.expander))
        next(w for w in at.selectbox if w.label=='Набор данных').select('MUF · временной ряд').run(timeout=60)
        self.assertFalse(at.exception)
        self.assertFalse(any(w.label=='Колонки даты / времени → datetime' for w in at.multiselect))
        for label,value in [('Строк для обучения',100),('Строк для валидации',30),('Строк для теста',20)]:
            next(w for w in at.number_input if w.label==label).set_value(value).run(timeout=60)
        next(w for w in at.button if w.label=='▶ Обучить и оценить').click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(at.session_state.result['sizes'],{'train':100,'validation':30,'test':20})


if __name__=='__main__': unittest.main()

class DatetimeTrainingTests(unittest.TestCase):
    def test_datetime_survives_both_tasks_and_missing_targets(self):
        dates=pd.date_range('2024-01-01',periods=120,freq='h')
        for task in ['Регрессия','Классификация']:
            with self.subTest(task=task):
                raw=pd.DataFrame({'datetime':dates,'x':np.arange(120),
                    'y':np.arange(120,dtype=float) if task=='Регрессия' else np.tile([0.,1.],60)})
                raw.loc[10,'y']=np.nan
                frame=prepare_frame(raw,target='y')
                X,y,parts=split_data(frame,['x'],'y',task,{'split':'time'})
                self.assertIsInstance(X.index,pd.DatetimeIndex)
                pd.testing.assert_index_equal(X.index,y.index)
                self.assertNotIn(dates[10],X.index)
                self.assertEqual(X.index.name,'datetime')
                result=run_experiments(frame,['x'],'y',task,[{'name':'Random Forest','params':{'trees':10}}],{'split':'time'})
                self.assertNotIn('error',result)
                self.assertIsInstance(result['predictions'].index,pd.DatetimeIndex)
                self.assertIsInstance(result['comparison'].index,pd.DatetimeIndex)
                pd.testing.assert_index_equal(result['predictions'].index,
                    pd.DatetimeIndex(result['predictions']['Время'],name='datetime'))
