import unittest
import numpy as np
import pandas as pd
from ml_lab.core import run_experiments, daily_metrics
from ml_lab.data import prepare_frame
from ml_lab.diagnostics import timeline_figure,daily_figure,regression_figures


class ResultPanelTests(unittest.TestCase):
    def test_train_test_timeline_and_metrics_have_correct_values(self):
        raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=240,freq='h'),'x':np.arange(240,dtype=float),'y':np.arange(240,dtype=float)*2+1})
        frame=prepare_frame(raw,target='y')
        result=run_experiments(frame,['x'],'y','Регрессия',[{'name':'Ridge','params':{}}],{'split':'time'})
        training=result['train_predictions']
        self.assertEqual(len(training),result['sizes']['train'])
        self.assertEqual(training.index.name,'datetime')
        np.testing.assert_allclose(training['Базовый прогноз'].iloc[1:],training['Факт'].iloc[1:]-2)
        figure=timeline_figure(result)
        self.assertEqual(len(figure.data),6)
        self.assertEqual(figure.data[0].x[0],training.index[0])
        self.assertEqual(figure.data[3].x[0],result['predictions'].index[0])
        self.assertNotIn('fit',regression_figures(result['predictions']))
        metrics=daily_figure(result)
        self.assertEqual([t.type for t in metrics.data],['bar','bar','bar','scatter']*2)
        np.testing.assert_allclose(metrics.data[3].y,daily_metrics(training,'Регрессия')['MAPE, %'])
        self.assertEqual(metrics.data[3].yaxis,'y3')
        self.assertEqual(metrics.data[7].yaxis,'y4')

    def test_training_predictions_do_not_change_recurrent_test_context(self):
        raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=100,freq='h'),'x':np.sin(np.arange(100)/10),'y':np.sin(np.arange(100)/10)})
        frame=prepare_frame(raw,target='y')
        result=run_experiments(frame,['x'],'y','Регрессия',[{'name':'LSTM','params':{'epochs':2,'units':8,'window':3}}],{'split':'time'})
        self.assertNotIn('error',result)
        predicted=result['pipeline'].predict(frame[['x']].iloc[-20:])
        np.testing.assert_allclose(predicted,result['predictions']['Прогноз'],rtol=1e-6)

    def test_classification_daily_bars_and_majority_baseline(self):
        raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=120,freq='h'),'x':np.tile([0,1],60),'y':np.tile(['a','b'],60)})
        frame=prepare_frame(raw,target='y')
        result=run_experiments(frame,['x'],'y','Классификация',[{'name':'Random Forest','params':{'trees':10}}],{'split':'time'})
        self.assertEqual(set(result['predictions']['Базовый прогноз']),{'a'})
        self.assertEqual(len(timeline_figure(result).data),6)
        self.assertEqual([t.type for t in daily_figure(result).data],['bar']*4)
