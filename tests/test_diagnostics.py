import unittest
import numpy as np
import pandas as pd
from ml_lab.diagnostics import regression_figures, probability_figures, shap_local_figure, daily_figure, comparison_figures, classification_figures


class DiagnosticsTests(unittest.TestCase):
    def test_cumulative_metrics_use_all_previous_errors_in_datetime_order(self):
        index=pd.date_range('2024-01-01',periods=3,freq='h')
        data=pd.DataFrame({'Факт':[1.,3.,6.],'Прогноз':[0.,1.,3.]},index=index)
        figures=regression_figures(data.iloc[::-1])
        curves={trace.name:trace for trace in figures['cumulative'].data}
        np.testing.assert_allclose(curves['MAE'].y,[1,1.5,2])
        np.testing.assert_allclose(curves['RMSE'].y,np.sqrt([1,2.5,14/3]))
        self.assertEqual(list(curves['MAE'].x),list(index))

    def test_class_metrics_allow_reserved_names_and_missing_predictions(self):
        result=dict(predictions=pd.DataFrame({'Факт':['accuracy','macro avg'],
            'Прогноз':['accuracy','accuracy']},index=pd.date_range('2024-01-01',periods=2,freq='h')))
        figures=classification_figures(result)
        np.testing.assert_allclose(figures['classes'].data[0].z,[[.5,1.,2/3],[0.,0.,0.]])
        np.testing.assert_allclose(figures['normalized'].data[0].z,[[1.,0.],[1.,0.]])

    def test_probability_curves_respect_probability_class_order(self):
        result=dict(probability_classes=['yes','no'],test_probabilities=np.array([[.9,.1],[.1,.9],[.8,.2],[.2,.8]]),
            predictions=pd.DataFrame({'Факт':['yes','no','yes','no']}))
        figures=probability_figures(result,'yes')
        self.assertIn('AUC=1.000',figures['roc'].layout.title.text)
        self.assertIn('AP=1.000',figures['pr'].layout.title.text)
        result['predictions']['Факт']='yes'
        self.assertEqual(probability_figures(result,'yes'),{})

    def test_waterfall_preserves_all_contributions_when_truncated(self):
        values=np.arange(-10,10,dtype=float).reshape(1,-1)
        explanation=dict(values=values,names=[str(i) for i in range(20)],bases=np.array([12.]),output='y')
        figure=shap_local_figure(explanation,0)
        self.assertAlmostEqual(sum(figure.data[0].y),12+values.sum())
        self.assertIn('Остальные признаки',figure.data[0].x)

    def test_undefined_daily_r2_and_failed_search_candidates(self):
        predictions=pd.DataFrame({'Время':pd.date_range('2024-01-01',periods=2,freq='D'),'Факт':[1.,2.],'Прогноз':[2.,2.]})
        result=dict(task='Регрессия',predictions=predictions,leaderboard=[
            {'Вариант':1,'Модель':'Ridge','Статус':'Готово','R²':.3,'Секунды':1},
            {'Вариант':2,'Модель':'Ridge','Статус':'error','R²':np.nan,'Секунды':0},
            {'Вариант':3,'Модель':'Ridge','Статус':'Готово','R²':.2,'Секунды':2}])
        self.assertIsNotNone(daily_figure(result))
        search=comparison_figures(result)['search']
        np.testing.assert_allclose(search.data[1].y,[.3,.3])
        self.assertEqual(list(search.data[0].x),[1,3])

class ExplanationInterfaceTests(unittest.TestCase):
    def test_explanation_survives_widget_rerun(self):
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
        next(w for w in at.selectbox if w.label=='Набор данных').select('MUF · временной ряд').run()
        next(w for w in at.selectbox if w.label=='Модель').select('Ridge').run()
        next(w for w in at.button if w.label=='▶ Обучить и оценить').click().run(timeout=60)
        next(w for w in at.button if w.label=='Рассчитать объяснение').click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertIn('Permutation Importance:0',at.session_state.result['explanations'])
        next(w for w in at.selectbox if w.label=='Метод').select('SHAP').run()
        next(w for w in at.selectbox if w.label=='Метод').select('Permutation Importance').run()
        self.assertFalse(at.exception)
        self.assertIn('Permutation Importance:0',at.session_state.result['explanations'])

class DecisionScoreExplanationTests(unittest.TestCase):
    def test_binary_shap_explains_the_selected_class_score(self):
        from ml_lab.core import run_experiments
        from ml_lab.data import prepare_frame
        from ml_lab.explain import shap_scores
        raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=80,freq='h'),
            'x':np.tile([-2.,-1.,1.,2.],20),'y':np.tile(['a','a','b','b'],20)})
        frame=prepare_frame(raw,target='y')
        result=run_experiments(frame,['x'],'y','Классификация',[{'name':'SVC','params':{}}],{'split':'time'})
        explanation=shap_scores(result,count=1,class_index=0)
        score=result['pipeline'].decision_function(result['explain_X'].head(1))[0]
        self.assertAlmostEqual(float(explanation['bases'][0]+explanation['values'][0].sum()),-score,places=5)
