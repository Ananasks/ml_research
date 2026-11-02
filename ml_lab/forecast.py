"""Date-based predictions beyond the training dataset, with explicit input requirements."""
import copy
import numpy as np
import pandas as pd
from ml_lab.data import prepare_frame, parse_datetime_values
from ml_lab.neural import RecurrentRegressor


def forecast_dates(result, raw_history, start, end, frequency='30min', future_inputs=None):
    config=result['settings']
    if not config['time_columns'] and not isinstance(raw_history.index,pd.DatetimeIndex): raise ValueError('Для прогноза по дате нужна колонка datetime при обучении.')
    if pd.Timestamp(end)<pd.Timestamp(start): raise ValueError('Конец интервала раньше начала.')
    if (pd.Timestamp(end)-pd.Timestamp(start))/pd.Timedelta(frequency)>=5000:
        raise ValueError('Выберите интервал не более 5000 точек.')
    dates=pd.date_range(start,end,freq=frequency)
    if not len(dates) or len(dates)>5000: raise ValueError('Выберите интервал от 1 до 5000 точек.')
    # Always forecast with a private copy so diagnostic prediction cannot alter the exported context.
    pipeline=copy.deepcopy(result['pipeline'])
    history=prepare_frame(raw_history,**config)
    target=config.get('target')
    target_history=[name for name,source in history.attrs.get('history_sources',{}).items() if source==target]
    if set(result['features']) & set(target_history):
        raise ValueError('Для будущего интервала без фактических Y лаги и другие признаки истории цели недоступны. Обучите модель на календарных/известных будущих признаках без истории Y.')
    last=history.attrs['time'].max()
    if dates[0]<=last:
        raise ValueError('Будущий прогноз должен начинаться после последней даты исходного набора. Для дат внутри набора используйте выбор периода на вкладке «Данные».')
    h=int(config['horizon'])
    if h:
        steps=history.attrs['time'].diff().dropna().unique()
        if len(steps)!=1: raise ValueError('При горизонте в строках будущий прогноз поддерживает только регулярный временной ряд.')
        delta=pd.Timedelta(steps[0])
        if pd.Timedelta(frequency)!=delta: raise ValueError('Шаг будущего прогноза должен совпадать с шагом обучающего ряда при ненулевом горизонте.')
    else: delta=pd.Timedelta(frequency)
    feature_dates=dates-h*delta
    base_settings=config|dict(calendar=False,repair=None,lags=(),rolling_windows=(),difference=False,
        ewm_spans=(),difference_lags=(),quality_flags=False,horizon=0)
    base=prepare_frame(raw_history,**base_settings)
    base.attrs={}
    if base.index.duplicated().any(): raise ValueError('Для будущего прогноза нужны уникальные datetime истории.')
    required=set()
    sources=config.get('feature_columns')
    if sources is None: sources=[target] if target else []
    for feature in result['features']:
        if feature.startswith('calendar_'): continue
        if feature in base.columns: required.add(feature);continue
        origin=next((column for column in sorted(base.columns,key=len,reverse=True)
            if feature in (column+'_was_missing',column+'_was_anomaly') or
            (column in sources and any(feature.startswith(column+suffix) for suffix in ('_lag_','_rolling_','_ewm_','_past_difference')))),None)
        required.add(origin or feature)
    if future_inputs is None:
        if required:
            raise ValueError('Модели нужны будущие значения X. Загрузите CSV/Excel с datetime и исходными входными колонками или обучите модель только на календарных признаках.')
        future=pd.DataFrame(index=pd.DatetimeIndex(feature_dates,name='datetime'))
    else:
        future=future_inputs.copy()
        if 'datetime' not in future: raise ValueError('В файле будущих признаков нужна колонка datetime.')
        future['datetime']=parse_datetime_values(future['datetime'])
        if future.datetime.isna().any() or future.datetime.duplicated().any(): raise ValueError('Даты будущих признаков должны быть корректными и уникальными.')
        future=future.set_index('datetime').reindex(feature_dates)
        if any(column not in future for column in required):
            raise ValueError('Для будущего прогноза нужны исходные X, включая источники лагов и скользящих окон.')
        repair=config.get('repair') or {}
        for column in required:
            if future[column].isna().any() and (column not in repair.get('columns',[]) or repair.get('fill_method') in (None,'none')):
                raise ValueError('Для каждой даты нужны все выбранные будущие входные значения без пропусков или включённое восстановление этих колонок.')
    # Rebuild exactly the same causal features and repair using actual history as context.
    # Historical observations win when the horizon refers to already known X dates.
    incoming=future.loc[~future.index.isin(base.index)].copy()
    incoming.attrs={}
    combined=pd.concat([base,incoming]).sort_index(kind='stable').rename_axis('datetime').reset_index()
    future_settings=config|dict(time_columns=['datetime'],horizon=0,frequency=None)
    prepared=prepare_frame(combined,**future_settings)
    future=prepared.reindex(feature_dates)
    if any(column not in future for column in result['features']):
        raise ValueError('Не удалось построить все будущие признаки.')
    model=pipeline.named_steps['model']
    if isinstance(model,RecurrentRegressor):
        model.prediction_breaks_=None
        steps=history.attrs['time'].diff().dropna().unique()
        if len(steps)!=1 or pd.Timedelta(steps[0])!=pd.Timedelta(frequency):
            raise ValueError('Будущий прогноз рекуррентной сети требует регулярного ряда и того же временного шага, что при обучении.')
        # Recurrent windows cannot skip unobserved feature rows in the gap.
        if feature_dates[0]-last>delta: raise ValueError('Для рекуррентной модели начните прогноз со следующего шага после исходного ряда; окна не должны пропускать неизвестный промежуток.')
        if model.window>1:
            preceding=history.loc[history.index<feature_dates[0],result['features']].tail(model.window-1)
            if len(preceding)<model.window-1: raise ValueError('Недостаточно исторических строк для окна прогноза.')
            model.context_=np.asarray(pipeline.named_steps['preprocess'].transform(preceding),dtype=np.float32)
    pred=pipeline.predict(future[result['features']])
    return pd.DataFrame({'datetime':dates,'Прогноз':pred},index=pd.DatetimeIndex(dates,name='datetime'))
