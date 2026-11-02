import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from ml_lab.data import prepare_frame
from ml_lab.core import run_experiments, split_data, generate_candidates, daily_metrics
from ml_lab.model_registry import MODEL_REGISTRY, sample_parameters, online_catalog_status
from ml_lab.explain import shap_scores, permutation_scores, integrated_gradients
from ml_lab.forecast import forecast_dates


class AdvancedTests(unittest.TestCase):
    def setUp(self):
        self.raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=240,freq='h'), 'x':np.sin(np.arange(240)/4)})
        self.raw['y']=self.raw.x*3+np.arange(240)*.01
        self.frame=prepare_frame(self.raw,['datetime'],calendar=True,target='y')

    def result(self,name='Ridge',features=None,params=None):
        features=features or ['x','calendar_hour']
        params=params or {'split':'time'}
        result=run_experiments(self.frame,features,'y','Регрессия',[{'name':name,'params':params}],params)
        self.assertNotIn('error',result,str(result))
        result['settings']={'time_columns':['datetime'],'calendar':True,'target':'y','horizon':0,'lags':[]}
        return result

    def test_datetime_column_is_typed(self):
        frame=prepare_frame(self.raw.astype({'datetime':str}),['datetime'],target='y')
        self.assertTrue(isinstance(frame.index,pd.DatetimeIndex))
        zoned=pd.DataFrame({'datetime':['2024-01-01T03:00:00+03:00','2024-01-01T01:00:00Z'],'y':[1,2]})
        converted=prepare_frame(zoned,['datetime'],target='y')
        self.assertEqual(converted.index[0],pd.Timestamp('2024-01-01 00:00:00'))

    def test_selected_date_interval_and_purge(self):
        frame=prepare_frame(self.raw,['datetime'],calendar=True,target='y',horizon=3)
        p={'split':'time','forecast_start':'2024-01-08T00:00:00','forecast_end':'2024-01-08T23:00:00'}
        X,y,(tr,va,te)=split_data(frame,['x'],'y','Регрессия',p)
        target_time=pd.Series(frame.attrs['time'].shift(-3).to_numpy(),index=frame.index)
        self.assertTrue((target_time.loc[X.iloc[te].index].dt.day==8).all())
        self.assertLess(target_time.loc[X.iloc[tr].index].max(),X.iloc[va].index.min())
        self.assertLess(target_time.loc[X.iloc[va].index].max(),X.iloc[te].index.min())
        r=run_experiments(frame,['x'],'y','Регрессия',[{'name':'Ridge','params':p}],p)
        self.assertEqual(len(r['predictions']),24)
        self.assertEqual(len(daily_metrics(r['predictions'],'Регрессия')),1)

    def test_single_timestamp_forecast(self):
        p={'split':'time','forecast_start':'2024-01-08T12:00:00','forecast_end':'2024-01-08T12:00:00'}
        r=self.result(params=p)
        self.assertEqual(len(r['predictions']),1)
        self.assertTrue(np.isnan(r['metrics']['R²']))

    def test_search_ranges_are_respected(self):
        spaces={'Ridge':{'alpha':dict(type='float',min=.2,max=.3,log=False)}}
        c=generate_candidates('Регрессия',['Ridge'],6,{'seed':123},spaces)
        self.assertEqual(len(c),6)
        self.assertTrue(all(.2<=v['params']['alpha']<=.3 for v in c))
        fixed={'alpha':dict(type='float',min=.2,max=.3,log=False),'__fixed__':{'max_iter':25}}
        p=sample_parameters('Ridge','Регрессия',{},fixed,rng=np.random.default_rng(1))
        self.assertEqual(p['max_iter'],25)

    def test_optuna_validation_selection(self):
        schema={'alpha':dict(type='float',min=.01,max=1.,log=True)}
        r=run_experiments(self.frame,['x'],'y','Регрессия',[],{'split':'time','seed':4},search={'families':['Ridge'],'spaces':{'Ridge':schema},'count':5})
        self.assertNotIn('error',r,str(r))
        self.assertEqual(len(r['leaderboard']),5)
        self.assertEqual(r['variant'],int(np.argmax([v['R²'] for v in r['leaderboard']]))+1)
        self.assertEqual(r['comparison'].groupby('Модель').size().nunique(),1)

    def test_shap_additivity_and_permutation(self):
        r=self.result()
        e=shap_scores(r,count=3)
        pred=r['pipeline'].predict(r['explain_X'].head(3))
        np.testing.assert_allclose(e['bases']+e['values'].sum(axis=1),pred,rtol=1e-6,atol=1e-6)
        self.assertEqual(set(permutation_scores(r)['Признак']),set(r['features']))

    def test_classification_shap_probabilities(self):
        frame=self.frame.copy();frame.y=np.where(frame.x>0,'a','b')
        r=run_experiments(frame,['x'],'y','Классификация',[{'name':'Logistic Regression','params':{}}],{'split':'random'})
        e=shap_scores(r,count=2,class_index=1)
        proba=r['pipeline'].predict_proba(r['explain_X'].head(2))[:,1]
        np.testing.assert_allclose(e['bases']+e['values'].sum(axis=1),proba,atol=1e-6)

    def test_future_calendar_and_required_covariates(self):
        r=self.result(features=['calendar_hour','calendar_hour_sin'])
        future=forecast_dates(r,self.raw,'2024-01-11','2024-01-11 03:00:00','1h')
        self.assertEqual(len(future),4)
        self.assertTrue(np.isfinite(future['Прогноз']).all())
        r=self.result()
        with self.assertRaisesRegex(ValueError,'будущие значения X'): forecast_dates(r,self.raw,'2024-01-11','2024-01-11 03:00:00','1h')
        inputs=pd.DataFrame({'datetime':pd.date_range('2024-01-11',periods=4,freq='h'),'x':range(4)})
        self.assertEqual(len(forecast_dates(r,self.raw,'2024-01-11','2024-01-11 03:00:00','1h',inputs)),4)

    def test_gru_rnn_and_gradients(self):
        params={'epochs':2,'units':8,'dense_units':8,'window':4,'learning_rate':.001,'batch_size':32,'split':'time'}
        for name in ['GRU','RNN']:
            with self.subTest(name=name):
                r=self.result(name,params=params)
                before=r['pipeline'].named_steps['model'].context_.copy()
                attribution=integrated_gradients(r,steps=3)
                self.assertTrue(np.isfinite(attribution['Важность']).all())
                np.testing.assert_array_equal(before,r['pipeline'].named_steps['model'].context_)

    def test_online_metadata_is_read_only(self):
        response=type('Response',(),{'raise_for_status':lambda self:None,'json':lambda self:{'info':{'version':'test'}}})()
        with patch('ml_lab.model_registry.requests.get',return_value=response) as get:
            rows=online_catalog_status()
        self.assertEqual(len(rows),2)
        self.assertEqual(get.call_count,2)
        self.assertTrue(all(call.args[0].startswith('https://pypi.org/pypi/') for call in get.call_args_list))
        import requests
        with patch('ml_lab.model_registry.requests.get',side_effect=requests.Timeout()):
            offline=online_catalog_status()
        self.assertEqual(len(offline),2)
        self.assertTrue(all('недоступен' in row['Статус'] for row in offline))


if __name__=='__main__': unittest.main()
