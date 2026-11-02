import io
import unittest
import numpy as np
import pandas as pd
import joblib
from ml_lab.core import MODELS, DEFAULTS, run_experiments, split_data, generate_candidates
from ml_lab.data import prepare_frame, read_dataset
from ml_lab.model_registry import RECURRENT_MODELS


class LabTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.df = pd.DataFrame({'x': rng.normal(size=120), 'category': ['a','b']*60})
        self.df['y'] = 3*self.df.x + rng.normal(0,.1,120)

    def test_read_csv_and_tsv(self):
        for sep in [',',';','\t']:
            frame = read_dataset(self.df.to_csv(index=False,sep=sep).encode(), 'data.csv')
            self.assertEqual(frame.shape,self.df.shape)
        with self.assertRaises(ValueError): read_dataset(b'only\n1', 'bad.csv')
        placeholders = read_dataset(b'x,y\n---,1\n2,3\n', 'data.csv')
        self.assertTrue(np.isnan(placeholders.loc[0,'x']))
        self.assertTrue(pd.api.types.is_numeric_dtype(placeholders.x))

    def test_time_features_and_horizon_purge(self):
        df = pd.DataFrame({'datetime':pd.date_range('2023-01-01',periods=120,freq='30min'), 'y':np.arange(120,dtype=float)})
        frame = prepare_frame(df, ['datetime'], calendar=True, target='y',lags=[1], horizon=3)
        self.assertEqual(frame.iloc[1]['y_lag_1'],0)
        self.assertEqual(frame.iloc[1]['y'],4)
        X,y,(tr,va,te) = split_data(frame,['calendar_hour','y_lag_1'],'y','Регрессия',{})
        self.assertLess(X.index[tr[-1]]+pd.Timedelta(minutes=90), X.index[va[0]])
        self.assertLess(X.index[va[-1]]+pd.Timedelta(minutes=90), X.index[te[0]])
        with self.assertRaises(ValueError): split_data(frame,['y_lag_1'],'y','Регрессия',{'split':'random'})

    def test_invalid_dates_are_explicitly_reported(self):
        df = pd.DataFrame({'datetime':['2023-01-01','30:00','2023-01-02'], 'y':[1,2,3]})
        with self.assertRaises(ValueError): prepare_frame(df,['datetime'],target='y')
        frame = prepare_frame(df,['datetime'],target='y',drop_invalid_time=True)
        self.assertEqual(len(frame),2)
        self.assertEqual(frame.attrs['invalid_time_rows'],1)

    def test_dayfirst_and_causal_lags(self):
        df = pd.DataFrame({'date':['02-01-2022','01-02-2022'], 'time':['00:00:00','12:30:00'], 'y':[5.,10.]})
        frame = prepare_frame(df,['date','time'],dayfirst=True,target='y',lags=[1])
        self.assertEqual(frame.attrs['time'].iloc[0].month,1)
        self.assertEqual(frame.attrs['time'].iloc[0].day,2)
        self.assertTrue(np.isnan(frame.iloc[0]['y_lag_1']))
        self.assertEqual(frame.iloc[1]['y_lag_1'],5.)

    def test_duplicate_timestamps_stay_together(self):
        df = self.df.copy()
        df['datetime'] = np.repeat(pd.date_range('2023-01-01',periods=40),3)
        frame = prepare_frame(df,['datetime'],target='y')
        X,y,(tr,va,te) = split_data(frame,['x'],'y','Регрессия',{'test_size':.22,'validation_size':.22})
        times = frame.attrs['time']
        self.assertLess(X.index[tr[-1]],X.index[va[0]])
        self.assertLess(X.index[va[-1]],X.index[te[0]])

    def test_all_classical_models(self):
        for task in MODELS:
            frame = self.df.copy()
            if task == 'Классификация': frame['y'] = np.where(frame.x > 0,'high','low')
            for name in MODELS[task]:
                if name in RECURRENT_MODELS: continue
                with self.subTest(task=task, name=name):
                    p = DEFAULTS | dict(split='random',iterations=10,trees=10,hidden_layers=[8],learning_rate=.001)
                    if name == 'Huber Regression': p['epsilon'] = 1.35
                    result = run_experiments(frame,['x','category'],'y',task,[{'name':name,'params':p}],p)
                    self.assertNotIn('error',result,str(result))
                    self.assertTrue(all(np.isfinite(v) for v in result['metrics'].values()))
                    self.assertEqual(len(result['predictions']),24)

    def test_selection_and_serialization(self):
        candidates = generate_candidates('Регрессия',['Linear Regression','Ridge'],5,{'split':'random'})
        self.assertEqual(candidates,generate_candidates('Регрессия',['Linear Regression','Ridge'],5,{'split':'random'}))
        result = run_experiments(self.df,['x'],'y','Регрессия',candidates,{'split':'random'})
        scores = [r['R²'] for r in result['leaderboard']]
        self.assertEqual(result['variant'],int(np.argmax(scores))+1)
        buf = io.BytesIO(); joblib.dump(result['pipeline'],buf); buf.seek(0)
        restored = joblib.load(buf)
        np.testing.assert_allclose(restored.predict(self.df[['x']]),result['pipeline'].predict(self.df[['x']]))

    def test_preprocessing_is_fit_on_train_only(self):
        frame = self.df.copy()
        frame.loc[72:,'x'] = 10000
        result = run_experiments(frame,['x'],'y','Регрессия',[{'name':'Linear Regression','params':{}}],{})
        scaler = result['pipeline'].named_steps['preprocess'].named_transformers_['num'].named_steps['scale']
        self.assertLess(scaler.mean_[0],1)

    def test_lstm_real_sequence_and_export(self):
        p = DEFAULTS | dict(epochs=2,window=4,units=8,dense_units=8,batch_size=16,learning_rate=.001)
        result = run_experiments(self.df,['x'],'y','Регрессия',[{'name':'LSTM','params':p}],p)
        self.assertNotIn('error',result,str(result))
        self.assertEqual(len(result['predictions']),24)
        model = result['pipeline'].named_steps['model']
        self.assertEqual(model.rnn_.input_size,1)
        self.assertEqual(model._windows(np.zeros((6,1)),np.zeros((3,1))).shape,(6,4,1))
        buf = io.BytesIO(); joblib.dump(result,buf); buf.seek(0)
        restored = joblib.load(buf)
        np.testing.assert_allclose(restored['pipeline'].predict(self.df[['x']].tail(24)),result['pipeline'].predict(self.df[['x']].tail(24)))


if __name__ == '__main__': unittest.main()
