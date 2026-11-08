"""Portable exports of computed results, diagnostics, explanations and chart styles."""
import io
import json
import zipfile
import pandas as pd
import plotly.express as px
import joblib
from ml_lab.experiments import clean,html_report
from ml_lab.core import daily_metrics
from ml_lab.diagnostics import (timeline_figure,daily_figure,regression_figures,classification_figures,
    comparison_figures,probability_figures,training_figure,importance_figure,shap_local_figure)
from ml_lab.chart_style import apply_style
from ml_lab.result_notes import NOTES
from ml_lab.model_theory import theory,architecture_dot,fitted_tree_dot


def results_bundle(result,styles=None,include_model=True):
    styles=styles or {}
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        def json_file(name,value): archive.writestr(name,json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False))
        def csv_file(name,frame): archive.writestr(name,frame.to_csv(index=True).encode('utf-8-sig'))
        json_file('metrics.json',{'train':result.get('train_metrics'),'validation':result.get('validation'),
            'test':result['metrics'],'evaluation_sizes':result.get('evaluation_sizes'),'sizes':result['sizes']})
        csv_file('metrics.csv',pd.DataFrame([dict(Выборка=split,**metrics) for split,metrics in [
            ('train',result.get('train_metrics',{})),('validation',result.get('validation',{})),('test',result['metrics'])]]))
        json_file('experiment.json',result['config'])
        json_file('display_settings.json',dict(probability_class=result.get('display_probability_class'),explanations=result.get('explanation_views',{})))
        json_file('chart_styles.json',styles)
        archive.writestr('report.html',html_report(result,result['config']))
        description=theory(result['name'],result['task'])
        archive.writestr('model_description.md',result['name']+'\n\n'+description['text']+'\n\n'+description['formula'])
        archive.writestr('charts/model_architecture.dot',architecture_dot(result['name'],result['task'],result.get('params',{})))
        if 'pipeline' in result:
            tree=fitted_tree_dot(result)
            if tree: archive.writestr('charts/fitted_tree.dot',tree)
        for name,key in [('test','predictions'),('train','train_predictions'),('validation_candidates','comparison')]:
            if key in result:
                csv_file(name+'_predictions.csv',result[key])
                if name in ('test','train'): csv_file(name+'_daily_metrics.csv',daily_metrics(result[key],result['task']))
        csv_file('leaderboard.csv',pd.DataFrame(result['leaderboard']))
        figures={'train_test_timeline':timeline_figure(result)}
        diagnostics=regression_figures(result['predictions']) if result['task']=='Регрессия' else classification_figures(result)
        figures.update({'quality_'+name:figure for name,figure in diagnostics.items()})
        figures.update({'comparison_'+name:figure for name,figure in comparison_figures(result).items()})
        if result.get('test_probabilities') is not None:
            selected=result.get('display_probability_class',result['probability_classes'][0])
            figures.update({'probability_'+name:figure for name,figure in probability_figures(result,selected).items()})
        daily=daily_figure(result)
        if daily is not None: figures['daily_metrics_train_test']=daily
        if 'pipeline' in result:
            loss=training_figure(result)
            if loss is not None: figures['training_curve']=loss
        for number,(label,payload) in enumerate(result.get('explanations',{}).items(),1):
            method=label.split(':')[0]
            table=payload['summary'] if method=='SHAP' else payload
            prefix=f'explanations/{number}'
            archive.writestr(prefix+'_method.txt',label)
            csv_file(prefix+'_summary.csv',table)
            figures[f'explanation_{number}']=importance_figure(table,method+' · validation')
            if method=='SHAP':
                index=pd.Index(payload['indices'],name='datetime')
                csv_file(prefix+'_shap.csv',pd.DataFrame(payload['values'],columns=payload['names'],index=index))
                csv_file(prefix+'_inputs.csv',pd.DataFrame(payload['inputs'],columns=payload['names'],index=index))
                json_file(prefix+'_bases.json',payload['bases'].tolist())
                view=result.get('explanation_views',{}).get(label,{})
                row=view.get('row',0)
                figures[f'explanation_{number}_local_selected']=shap_local_figure(payload,row)
                feature=view.get('feature',payload['names'][0]);column=payload['names'].index(feature)
                dependence=pd.DataFrame({'Значение признака':payload['inputs'][:,column],'SHAP-вклад':payload['values'][:,column]})
                figures[f'explanation_{number}_dependence']=px.scatter(dependence,x='Значение признака',y='SHAP-вклад',title='SHAP · '+feature)
        for key,figure in figures.items():
            style_key=('shap_local' if key.endswith('local_selected') else 'shap_dependence' if key.endswith('dependence') else 'importance') if key.startswith('explanation_') else key
            styled=apply_style(figure,styles.get(style_key,{}))
            # JSON preserves all trace values and formatting; the HTML is interactive offline.
            archive.writestr('charts/'+key+'.json',styled.to_json())
            archive.writestr('charts/'+key+'.html',styled.to_html(full_html=True,include_plotlyjs=True))
        json_file('chart_descriptions.json',NOTES)
        if include_model:
            model=io.BytesIO();joblib.dump(dict(pipeline=result['pipeline'],config=result['config']),model)
            archive.writestr('model.joblib',model.getvalue())
        archive.writestr('README.txt','ML Research\nМетрики test рассчитаны по исходным измерениям. Train — качество подгонки.\n'
            'ZIP содержит только уже рассчитанные объяснения; новые объяснения не запускаются.\n'
            'charts/*.html открываются без интернета; JSON хранит точные значения и оформление.\n'
            'Локальный SHAP соответствует выбранной строке (по умолчанию первой); все вклады сохранены в CSV.\n')
    return output.getvalue()
