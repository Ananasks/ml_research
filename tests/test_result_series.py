import unittest
import numpy as np
import pandas as pd
from ml_lab.core import run_experiments
from ml_lab.data import prepare_frame
from ml_lab.diagnostics import daily_metric_figures
from ml_lab.result_series import result_series, plot_records, winning_validation
from ml_lab.time_index import parse_datetime_values, normalize_time_index


class ResultSeriesTests(unittest.TestCase):
    def test_validation_contains_only_winner_and_covers_full_period(self):
        raw = pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=6000,freq='h'),
                            'x':np.arange(6000), 'y':3*np.arange(6000)+10})
        frame = prepare_frame(raw,target='y')
        result = run_experiments(frame,['x'],'y','Регрессия',[
            {'name':'Ridge','params':{'fit_intercept':False}},
            {'name':'Ridge','params':{'fit_intercept':True}}],{'split':'time'})
        series = result_series(result)
        self.assertEqual(len(winning_validation(result)),result['sizes']['validation'])
        self.assertEqual(len(series['splits']['train']),result['sizes']['train'])
        self.assertEqual(series['splits']['train'][-1]['Время'],result['train_predictions']['Время'].iloc[-1].isoformat()+'.000')
        labels={row['Модель'] for row in series['splits']['validation']}
        self.assertEqual(labels,{f"{result['variant']} · {result['name']}"})
        pd.testing.assert_series_equal(winning_validation(result)['Прогноз'],
            result['comparison'].loc[result['comparison']['Модель'].eq(f"{result['variant']} · {result['name']}")]['Прогноз'])

    def test_reduction_keeps_endpoints_extrema_and_gap_boundaries(self):
        times = pd.date_range('2024-01-01',periods=20000,freq='h').to_series(index=range(20000))
        times.iloc[10000:] += pd.Timedelta(days=30)
        frame=pd.DataFrame({'Время':times,'Факт':np.sin(np.arange(20000)), 'Прогноз':np.zeros(20000)})
        frame.loc[9789,'Факт']=999
        result=plot_records(frame,limit=300)
        self.assertLess(len(result),500)
        self.assertEqual(result[0]['Время'],frame['Время'].iloc[0].isoformat()+'.000')
        self.assertEqual(result[-1]['Время'],frame['Время'].iloc[-1].isoformat()+'.000')
        self.assertIn(999,[row['Факт'] for row in result])
        self.assertTrue(any(row['_gapBefore'] for row in result))

    def test_classification_labels_remain_strings(self):
        rows=plot_records(pd.DataFrame({'Время':pd.date_range('2024-01-01',periods=3),
                                       'Факт':['a','b','a'],'Прогноз':['b','b','a']}))
        self.assertEqual(rows[0]['Факт'],'a')

    def test_daily_metrics_have_independent_axes_and_dates(self):
        frame=pd.DataFrame({'Время':pd.date_range('2024-01-01',periods=72,freq='h'),
                            'Факт':np.arange(72)+1,'Прогноз':np.arange(72)+2})
        result={'task':'Регрессия','train_predictions':frame,'predictions':frame,'comparison':pd.DataFrame()}
        figures=daily_metric_figures(result)
        self.assertEqual(len(figures),8)
        for key,fig in figures.items():
            self.assertEqual(len(fig.data),1)
            self.assertEqual(fig.layout.xaxis.type,'date')
            self.assertEqual(fig.layout.yaxis.title.text,key.split(':')[-1])
            self.assertEqual(len(fig.data[0].x),3)


class ResearchDatetimeTests(unittest.TestCase):
    def test_iso_is_never_swapped_even_with_dayfirst_override(self):
        result=parse_datetime_values(pd.Series(['2024-02-03 10:20:00','2024-02-04T12:00:00+03:00']),dayfirst=True)
        self.assertEqual(result.iloc[0],pd.Timestamp('2024-02-03 10:20'))
        self.assertEqual(result.iloc[1],pd.Timestamp('2024-02-04 09:00'))

    def test_research_date_time_uses_one_convention_for_the_whole_column(self):
        raw=pd.DataFrame({'date':['02-01-2022','13-01-2022','03-02-2022'],
                          'time':['12:30:00','00:00:00','01:00:00'],'y':[1,2,3]})
        frame=prepare_frame(raw,target='y')
        self.assertEqual(frame.index[0],pd.Timestamp('2022-01-02 12:30'))
        self.assertEqual(frame.index[-1],pd.Timestamp('2022-02-03 01:00'))
        self.assertEqual(frame['calendar_month'].tolist(),[1,1,2])

    def test_each_builtin_research_dataset_has_correct_range(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]/'data'
        for name,start in [('muf.csv','2023-01-01'),('athens.csv','2021-06-01'),('tec_cb.csv','2022-01-01')]:
            with self.subTest(name=name):
                raw=pd.read_csv(root/name)
                _,time,diagnostics=normalize_time_index(raw,drop_invalid_time=True)
                self.assertTrue(time.is_monotonic_increasing)
                self.assertEqual(time.iloc[0].normalize(),pd.Timestamp(start))
                if name=='tec_cb.csv':
                    parsed=parse_datetime_values(raw.date+' '+raw.time)
                    self.assertEqual(parsed.iloc[48],pd.Timestamp('2022-01-03'))
                if name=='muf.csv':self.assertEqual(diagnostics['invalid_time_rows'],26)

    def test_date_only_and_case_whitespace_are_supported(self):
        frame=prepare_frame(pd.DataFrame({' Date ':['02-01-2022','03-01-2022'],'y':[1,2]}),target='y')
        self.assertEqual(frame.index[0],pd.Timestamp('2022-01-02'))
        self.assertEqual(frame.index[1],pd.Timestamp('2022-01-03'))


def test_complete_grid_predictions_retain_restored_targets_in_every_split():
    from ml_lab.core import run_experiments
    from ml_lab.data import prepare_frame
    from ml_lab.result_series import winning_validation,result_series
    raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=180,freq='h'),'y':10+np.sin(np.arange(180)/8)})
    raw=raw.drop(index=[40,41,42,120,121,122,160,161,162])
    prepared=prepare_frame(raw,target='y',frequency='auto',restoration={'columns':['y'],'auto_tune':False,'model_config':{'model':'ridge'}})
    result=run_experiments(prepared,['calendar_hour'],'y','Регрессия',[{'name':'Ridge','params':{}}],{'split':'time'})
    assert 'error' not in result
    for frame in [result['train_predictions'],winning_validation(result),result['predictions']]:
        assert 'Цель после восстановления' in frame
        assert frame['Цель после восстановления'].notna().all()
        assert frame['Прогноз'].notna().all()
        assert (frame['Время'].diff().dropna()==pd.Timedelta('1h')).all()
    assert result['predictions']['Факт'].isna().any()
    assert result['evaluation_sizes']['test']<result['sizes']['test']
    for rows in result_series(result)['splits'].values():
        assert not any(row.get('_gapBefore') for row in rows)
