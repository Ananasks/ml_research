import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ml_lab.chart_style import apply_style


class ChartStyleTests(unittest.TestCase):
    def test_style_preserves_datetime_and_values_and_original_figure(self):
        dates=pd.date_range('2024-01-01',periods=3,freq='h')
        original=px.line(x=dates,y=[1,3,2],labels={'x':'datetime','y':'y'})
        before=original.to_json()
        figure=apply_style(original,dict(title='Измерения',axes={'xaxis':'Дата','yaxis':'TEC'},grid='Только Y',
            grid_color='#112233',theme='Светлая',line_width=4,dash='Штриховая',mode='Линии и точки',
            custom_colors=True,colors={'0':'#ff0000'},legend=False))
        self.assertEqual(figure.layout.xaxis.title.text,'Дата')
        self.assertEqual(figure.layout.yaxis.title.text,'TEC')
        self.assertFalse(figure.layout.xaxis.showgrid)
        self.assertTrue(figure.layout.yaxis.showgrid)
        self.assertEqual(figure.data[0].line.color,'#ff0000')
        self.assertEqual(figure.data[0].line.dash,'dash')
        self.assertEqual(figure.data[0].mode,'lines+markers')
        np.testing.assert_array_equal(figure.data[0].x,original.data[0].x)
        np.testing.assert_array_equal(figure.data[0].y,original.data[0].y)
        self.assertEqual(original.to_json(),before)

    def test_heatmap_keeps_numeric_bounds_and_changes_scale(self):
        original=px.imshow([[0,1],[1,0]],zmin=0,zmax=1)
        figure=apply_style(original,dict(grid='Нет',scale='Cividis',axes={'xaxis':'Прогноз','yaxis':'Факт'}))
        self.assertEqual(figure.layout.coloraxis.cmin,0)
        self.assertEqual(figure.layout.coloraxis.cmax,1)
        self.assertNotEqual(figure.layout.coloraxis.colorscale,original.layout.coloraxis.colorscale)
        self.assertEqual(figure.layout.yaxis.autorange,original.layout.yaxis.autorange)

    def test_facet_labels_and_waterfall_semantics(self):
        original=go.Figure(go.Waterfall(x=['base','a','b','total'],y=[1,2,-1,0],measure=['absolute','relative','relative','total']))
        figure=apply_style(original,dict(custom_colors=True,positive_color='#00ff00',negative_color='#ff0000',total_color='#0000ff'))
        self.assertEqual(figure.data[0].increasing.marker.color,'#00ff00')
        self.assertEqual(figure.data[0].measure,original.data[0].measure)
        self.assertEqual(figure.data[0].y,original.data[0].y)
        facets=px.line(pd.DataFrame({'x':[1,2,1,2],'v':[2,3,4,5],'metric':['a','a','b','b']}),x='x',y='v',facet_row='metric')
        styled=apply_style(facets,dict(axes={'yaxis':'MAE','yaxis2':'RMSE'},grid='Обе оси'))
        self.assertEqual(styled.layout.yaxis2.title.text,'RMSE')
        self.assertTrue(styled.layout.xaxis2.showgrid)

    def test_editor_applies_independent_settings_and_resets(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
        prefix='chart_style_widget_source_chart__'
        at.text_input(key=prefix+'xaxis').set_value('Время измерения')
        at.text_input(key=prefix+'yaxis').set_value('Значение TEC')
        at.selectbox(key=prefix+'grid').select('Нет')
        at.button(key='FormSubmitter:chart_style_form_source_chart-Применить оформление').click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state['chart_styles']['source_chart']['axes']['yaxis'],'Значение TEC')
        self.assertNotIn('data_correlation',at.session_state['chart_styles'])
        at.button(key='FormSubmitter:chart_style_form_source_chart-Сбросить оформление').click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertNotIn('source_chart',at.session_state['chart_styles'])
        self.assertNotEqual(at.text_input(key=prefix+'xaxis').value,'Время измерения')
