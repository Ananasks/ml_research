"""Plotly diagnostics built from frozen experiment outputs, without refitting models."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_fscore_support, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from ml_lab.core import daily_metrics
from plotly.subplots import make_subplots


def regression_figures(predictions):
    data=predictions.sort_index(kind='stable').copy()
    data['Остаток']=data['Факт']-data['Прогноз']
    data['Абсолютная ошибка']=data['Остаток'].abs()
    figures={}
    residual=px.scatter(data,x='Прогноз',y='Остаток',title='Остатки против прогноза · test')
    residual.add_hline(y=0,line_dash='dash')
    figures['residual']=residual
    figures['distribution']=px.histogram(data,x='Остаток',nbins=40,title='Распределение остатков · test')
    figures['error_time']=px.line(data,x=data.index,y='Абсолютная ошибка',title='Абсолютная ошибка во времени · test')
    cumulative=pd.DataFrame(index=data.index)
    cumulative['MAE']=data['Абсолютная ошибка'].expanding().mean()
    cumulative['RMSE']=np.sqrt(data['Остаток'].pow(2).expanding().mean())
    figures['cumulative']=px.line(cumulative,x=cumulative.index,y=['MAE','RMSE'],title='Накопленные ошибки · test')
    # Row-lag correlation works without a regular grid; it is not a time-duration lag.
    max_lag=min(40,len(data)//4)
    if max_lag>=1 and data['Остаток'].nunique()>1:
        acf=pd.DataFrame({'Лаг · строк':range(1,max_lag+1),
            'Корреляция':[data['Остаток'].autocorr(lag) for lag in range(1,max_lag+1)]})
        figures['acf']=px.bar(acf,x='Лаг · строк',y='Корреляция',title='Автокорреляция остатков · лаг в наблюдениях')
    return figures


def timeline_figure(result):
    parts=[('Обучающая выборка · train',result.get('train_predictions')),('Тестовая выборка · test',result['predictions'])]
    parts=[(title,frame) for title,frame in parts if frame is not None]
    figure=make_subplots(rows=len(parts),cols=1,subplot_titles=[title for title,_ in parts],vertical_spacing=.16)
    colors={'Факт':'#2563eb','Прогноз':'#ea580c','Базовый прогноз':'#16a34a'}
    for row,(title,frame) in enumerate(parts,1):
        frame=frame.sort_index(kind='stable')
        for name,color in colors.items():
            if name not in frame: continue
            figure.add_trace(go.Scatter(x=frame.index,y=frame[name],name=name,legendgroup=name,showlegend=row==1,
                mode='lines' if result['task']=='Регрессия' else 'markers',connectgaps=False,
                line=dict(color=color,dash='dash' if name=='Базовый прогноз' else 'solid'),marker=dict(color=color)),row=row,col=1)
        figure.update_xaxes(title_text='datetime',row=row,col=1)
        figure.update_yaxes(title_text=result.get('target','Значение') if result['task']=='Регрессия' else 'Класс',row=row,col=1)
    figure.update_layout(title='Временной ход обучающей и тестовой выборки',height=650,hovermode='x unified',template='plotly_white',legend=dict(orientation='h',y=-.15))
    return figure


def daily_figure(result):
    parts=[('Train · качество на обучающих данных',result.get('train_predictions')),('Test · независимая оценка',result['predictions'])]
    tables=[(title,daily_metrics(frame,result['task'])) for title,frame in parts if frame is not None]
    tables=[(title,table) for title,table in tables if not table.empty]
    if not tables: return None
    regression=result['task']=='Регрессия'
    rows=2 if regression else 1
    labels=['MAE / RMSE / R²','MAPE, % · меньше — лучше'] if regression else ['Accuracy / F1 · ближе к 1 — лучше']
    figure=make_subplots(rows=rows,cols=len(tables),shared_xaxes=True,
        subplot_titles=[title+'<br>'+label for label in labels for title,_ in tables],
        horizontal_spacing=.12,vertical_spacing=.24 if regression else .1)
    colors={'MAE':'#2563eb','RMSE':'#f97316','R²':'#16a34a','MAPE, %':'#9333ea',
            'Accuracy':'#2563eb','F1 macro':'#16a34a'}
    for column,(_,table) in enumerate(tables,1):
        dates=table['День']
        metrics=['MAE','RMSE','R²','MAPE, %'] if regression else ['Accuracy','F1 macro']
        for metric in metrics:
            row=(2 if metric=='MAPE, %' else 1) if regression else 1
            common=dict(x=dates,y=table[metric],name=metric,legendgroup=metric,
                showlegend=column==1,customdata=table[['Измерений']].to_numpy(),
                hovertemplate='%{x|%d.%m.%Y}<br>'+metric+': %{y:.4g}<br>Измерений: %{customdata[0]}<extra></extra>')
            if metric=='MAPE, %':
                trace=go.Scatter(**common,mode='lines+markers',connectgaps=False,
                    line=dict(color=colors[metric],width=2.5),marker=dict(color=colors[metric],size=4))
            else:
                trace=go.Bar(**common,marker=dict(color=colors[metric]))
            figure.add_trace(trace,row=row,col=column)
        for row in range(1,rows+1):
            figure.update_yaxes(title_text=['MAE / RMSE / R²','MAPE, %'][row-1] if regression else 'Доля',
                row=row,col=column,showgrid=True,gridcolor='#e2e8f0',zerolinecolor='#94a3b8')
            figure.update_xaxes(title_text='Дата' if row==rows else None,type='date',nticks=4,
                tickformat='%d.%m<br>%Y',tickangle=0,showgrid=False,row=row,col=column)
        if regression:
            figure.add_hline(y=1,line_dash='dot',line_color='#86b79a',row=2,col=column)
        else:
            figure.update_yaxes(range=[0,1.05],row=1,col=column)
    figure.update_layout(title='Метрики по дням · обучение и тест',template='plotly_white',
        height=900 if regression else 500,hovermode='x unified',
        font=dict(family='Arial, sans-serif',size=13,color='#24344b'),
        legend=dict(orientation='h',y=-.18,x=0,title_text='Метрики'),
        margin=dict(t=100,b=150,l=80,r=45))
    return figure


def comparison_figures(result):
    table=pd.DataFrame(result['leaderboard'])
    metric='R²' if result['task']=='Регрессия' else 'F1 macro'
    if metric not in table: return {}
    table=table[table['Статус'].eq('Готово') & pd.to_numeric(table[metric],errors='coerce').notna()].copy()
    if table.empty: return {}
    table['Кандидат']=table['Вариант'].astype(str)+' · '+table['Модель']
    table=table.sort_values('Вариант')
    table['Лучшее значение']=table[metric].cummax()
    search=go.Figure()
    search.add_scatter(x=table['Вариант'],y=table[metric],mode='markers',name='Validation',text=table['Модель'])
    search.add_scatter(x=table['Вариант'],y=table['Лучшее значение'],mode='lines',name='Лучшее к этому шагу')
    search.update_layout(title='История поиска · validation',xaxis_title='Вариант',yaxis_title=metric)
    return dict(ranking=px.bar(table.sort_values(metric),x=metric,y='Кандидат',orientation='h',title='Качество кандидатов · validation'),
        search=search,cost=px.scatter(table,x='Секунды',y=metric,hover_name='Кандидат',title='Качество и время обучения · validation'))


def training_figure(result):
    model=result['pipeline'].named_steps['model']
    model=getattr(model,'regressor_',model)
    losses=getattr(model,'loss_curve_',None)
    if losses is None or not len(losses): return None
    figure=go.Figure()
    figure.add_scatter(x=list(range(1,len(losses)+1)),y=losses,name='Train loss',mode='lines')
    validation=getattr(model,'validation_curve_',None)
    if validation is not None:
        figure.add_scatter(x=list(range(1,len(validation)+1)),y=validation,name='Внутренняя validation loss',mode='lines')
        finite=np.asarray(validation,dtype=float)
        if np.isfinite(finite).any(): figure.add_vline(x=int(np.nanargmin(finite))+1,line_dash='dash')
    figure.update_layout(title='Кривая обучения победителя',xaxis_title='Эпоха / итерация',yaxis_title='Функция потерь')
    return figure


def classification_figures(result):
    data=result['predictions']; truth=data['Факт'].astype(str); predicted=data['Прогноз'].astype(str)
    labels=sorted(set(truth)|set(predicted))
    counts=pd.crosstab(truth,predicted).reindex(index=labels,columns=labels,fill_value=0)
    fractions=counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
    precision,recall,f1,_=precision_recall_fscore_support(truth,predicted,labels=labels,zero_division=0)
    per_class=pd.DataFrame({'precision':precision,'recall':recall,'f1-score':f1},index=labels)
    accuracy=(truth==predicted).astype(float).sort_index(kind='stable').expanding().mean()
    return dict(confusion=px.imshow(counts,text_auto=True,title='Матрица ошибок · test',labels={'x':'Прогноз','y':'Факт'}),
        normalized=px.imshow(fractions,zmin=0,zmax=1,text_auto='.2f',title='Матрица ошибок · доли внутри истинного класса'),
        classes=px.imshow(per_class,zmin=0,zmax=1,text_auto='.2f',title='Precision / Recall / F1 по классам · test'),
        accuracy=px.line(x=accuracy.index,y=accuracy.to_numpy(),title='Накопленная Accuracy · test',labels={'x':'datetime','y':'Accuracy'}))


def probability_figures(result, selected_class):
    scores=result.get('test_probabilities')
    if scores is None: return {}
    classes=result['probability_classes']
    actual=(result['predictions']['Факт'].astype(str).to_numpy()==str(selected_class)).astype(int)
    if len(np.unique(actual))<2: return {}
    probabilities=np.asarray(scores)[:,classes.index(selected_class)]
    fpr,tpr,_=roc_curve(actual,probabilities)
    precision,recall,_=precision_recall_curve(actual,probabilities)
    observed,estimated=calibration_curve(actual,probabilities,n_bins=min(10,len(actual)),strategy='quantile')
    roc=px.line(x=fpr,y=tpr,title=f'ROC · one-vs-rest · AUC={roc_auc_score(actual,probabilities):.3f}',labels={'x':'False positive rate','y':'True positive rate'})
    roc.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dash'))
    pr=px.line(x=recall,y=precision,title=f'Precision–Recall · AP={average_precision_score(actual,probabilities):.3f}',labels={'x':'Recall','y':'Precision'})
    pr.add_hline(y=float(actual.mean()),line_dash='dash')
    calibration=px.line(x=estimated,y=observed,markers=True,title='Калибровка вероятности · test',labels={'x':'Средняя предсказанная вероятность','y':'Доля положительных исходов'})
    calibration.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dash'))
    return dict(roc=roc,pr=pr,calibration=calibration)


def importance_figure(table, title):
    table=table.head(30).sort_values('Важность')
    return px.bar(table,x='Важность',y='Признак',orientation='h',
        error_x='Разброс' if 'Разброс' in table else None,title=title)


def shap_local_figure(explanation, row):
    values=np.asarray(explanation['values'][row]); names=np.asarray(explanation['names'])
    chosen=np.argsort(np.abs(values))[-15:][::-1]
    omitted=float(values.sum()-values[chosen].sum())
    contributions=values[chosen].tolist()+[omitted]
    labels=names[chosen].tolist()+['Остальные признаки']
    base=float(np.asarray(explanation['bases']).reshape(-1)[row])
    figure=go.Figure(go.Waterfall(measure=['absolute']+['relative']*len(contributions)+['total'],
        x=['Базовое значение']+labels+['Итог'],y=[base]+contributions+[0]))
    figure.update_layout(title='SHAP · вклад признаков в выбранный прогноз',yaxis_title=explanation['output'])
    return figure


def daily_metric_figures(result, basis='measured'):
    """One axis per metric/split avoids mixing units and subplot label collisions."""
    from ml_lab.result_series import winning_validation
    if basis not in ('measured','prepared'): raise ValueError('Неизвестная основа метрик.')
    figures = {}
    for split, frame in [('train', result.get('train_predictions')),
                         ('validation', winning_validation(result)), ('test', result.get('predictions'))]:
        if frame is None or frame.empty:
            continue
        if basis=='prepared':
            if 'Цель после восстановления' not in frame: continue
            frame=frame.copy()
            frame['Факт']=frame['Цель после восстановления']
        table = daily_metrics(frame, result['task'])
        if table.empty:
            continue
        for metric in (['MAE', 'RMSE', 'R²', 'MAPE, %'] if result['task']=='Регрессия' else ['Accuracy', 'F1 macro']):
            fig = go.Figure(go.Scatter(x=table['День'], y=table[metric], mode='lines+markers',
                connectgaps=False, name=metric, customdata=table[['Измерений']].to_numpy(),
                hovertemplate='%{x|%d.%m.%Y}<br>'+metric+': %{y:.4g}<br>Измерений: %{customdata[0]}<extra></extra>'))
            fig.update_layout(height=340, template='plotly_white', showlegend=False,
                margin=dict(l=65,r=25,t=25,b=70),
                xaxis=dict(title='Дата · UTC', type='date', nticks=6, tickformat='%d.%m<br>%Y', automargin=True),
                yaxis=dict(title=metric, automargin=True))
            figures[f"{'daily_prepared' if basis=='prepared' else 'daily'}:{split}:{metric}"] = fig
    return figures
