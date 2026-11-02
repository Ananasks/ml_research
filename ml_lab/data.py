"""Dataset readers and causal feature preparation, shared by UI and tests."""
import io
import csv
import numpy as np
import pandas as pd
from ml_lab.repair import repair_signals
from ml_lab.time_index import detect_time_columns, parse_datetime_values, normalize_time_index


def read_dataset(raw, name, separator=None, encoding='utf-8-sig', decimal='.', sheet=0):
    if name.lower().endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, na_values=['---', '--'])
    else:
        if separator is None:
            try:
                separator = csv.Sniffer().sniff(raw[:65536].decode(encoding), delimiters=',;\t|').delimiter
            except csv.Error:
                separator = ','
        df = pd.read_csv(io.BytesIO(raw), sep=separator, engine='python', encoding=encoding, decimal=decimal, na_values=['---', '--'])
    df.columns = df.columns.map(str)
    if df.empty or len(df.columns) < 2:
        raise ValueError('Нужны непустые данные и минимум две колонки.')
    if not df.columns.is_unique:
        raise ValueError('Названия колонок должны быть уникальными.')
    return df


def prepare_frame(df, time_columns=None, dayfirst=False, calendar=True, target=None, lags=(), horizon=0, drop_invalid_time=False, strict_dates=True, frequency=None,
                  duplicate_policy="error", rolling_windows=(), rolling_stats=("mean","std"), difference=False, start_datetime=None, index_frequency=None, repair=None,
                  feature_columns=None, ewm_spans=(), difference_lags=(), partial_windows=False, quality_flags=False):
    frame,time,diagnostics=normalize_time_index(df,time_columns,dayfirst,strict_dates,drop_invalid_time,
        frequency,duplicate_policy,start_datetime,index_frequency,target)
    frame=frame.replace([np.inf,-np.inf],np.nan)
    original_target=frame[target].copy() if target is not None and target in frame else None
    repair_records=[]; repair_masks={}; repair_originals={}
    if repair:
        frame,repair_records,repair_masks,repair_originals=repair_signals(frame,time,repair)
    repaired_values={column:frame[column].copy() for column in repair_originals}
    if calendar and time is not None:
        for name, values in {'hour': time.dt.hour, 'minute': time.dt.minute,
                             'dayofweek': time.dt.dayofweek, 'month': time.dt.month,
                             'dayofyear': time.dt.dayofyear, 'quarter': time.dt.quarter,
                             'day': time.dt.day, 'weekofyear': time.dt.isocalendar().week.astype(int)}.items():
            key = f'calendar_{name}'
            frame[key] = values
        hour=time.dt.hour+time.dt.minute/60+time.dt.second/3600
        frame['calendar_hour_sin'] = np.sin(2 * np.pi * hour / 24)
        frame['calendar_hour_cos'] = np.cos(2 * np.pi * hour / 24)
        frame['calendar_year_sin'] = np.sin(2*np.pi*(time.dt.dayofyear-1)/365.25)
        frame['calendar_year_cos'] = np.cos(2*np.pi*(time.dt.dayofyear-1)/365.25)
        frame['calendar_week_sin']=np.sin(2*np.pi*(time.dt.dayofweek+hour/24)/7)
        frame['calendar_week_cos']=np.cos(2*np.pi*(time.dt.dayofweek+hour/24)/7)
        frame['calendar_weekend']=(time.dt.dayofweek>=5).astype(int)
    history_columns=[]
    history_sources={}
    persistence=None
    if target is not None and target in frame:
        persistence=pd.to_numeric(frame[target],errors='coerce').shift(1)
    sources=([target] if target is not None else []) if feature_columns is None else feature_columns
    if (lags or rolling_windows or difference or ewm_spans or difference_lags) and not sources:
        raise ValueError('Выберите числовые колонки для признаков истории.')
    for source in sources:
        if source not in frame: raise ValueError(f'Колонка истории отсутствует: {source}')
        numeric_source=pd.to_numeric(frame[source],errors='coerce')
        past=numeric_source.shift(1)
        for lag in lags:
            if int(lag)<1: raise ValueError('Лаг должен быть положительным.')
            key=f'{source}_lag_{int(lag)}'
            frame[key]=numeric_source.shift(int(lag));history_columns.append(key);history_sources[key]=source
        for window in rolling_windows:
            if int(window)<2: raise ValueError('Скользящее окно должно содержать минимум 2 шага.')
            rolling=past.rolling(int(window),min_periods=1 if partial_windows else int(window))
            for stat in rolling_stats:
                if stat not in ('mean','std','min','max','median','iqr','trend'): raise ValueError('Неизвестная скользящая статистика.')
                key=f'{source}_rolling_{stat}_{int(window)}'
                if stat=='iqr': frame[key]=rolling.quantile(.75)-rolling.quantile(.25)
                elif stat=='trend':
                    def past_trend(values):
                        valid=np.flatnonzero(np.isfinite(values))
                        if not len(valid): return np.nan
                        if len(valid)==1: return 0.
                        return (values[valid[-1]]-values[valid[0]])/(valid[-1]-valid[0])
                    frame[key]=rolling.apply(past_trend,raw=True)
                elif stat=='std': frame[key]=rolling.std(ddof=0) if partial_windows else rolling.std()
                else: frame[key]=getattr(rolling,stat)()
                history_columns.append(key);history_sources[key]=source
        spans=[int(span) for span in ewm_spans]
        for span in spans:
            if span<2: raise ValueError('EWM span должен быть не меньше 2.')
            key=f'{source}_ewm_{span}'
            frame[key]=past.ewm(span=span,adjust=False,min_periods=1 if partial_windows else span).mean();history_columns.append(key);history_sources[key]=source
        differences=set(int(lag) for lag in difference_lags)
        if difference: differences.add(1)
        for lag in sorted(differences):
            if lag<1: raise ValueError('Лаг разности должен быть положительным.')
            key=f'{source}_past_difference' if lag==1 else f'{source}_past_difference_{lag}'
            frame[key]=past.diff(lag);history_columns.append(key);history_sources[key]=source
    if quality_flags:
        for column,masks in repair_masks.items():
            for label in ('missing','anomaly'):
                key=f'{column}_was_{label}'
                flag=pd.Series(masks[label],index=frame.index).astype(float)
                # A current target anomaly flag would disclose its value at h=0.
                frame[key]=flag.shift(1) if column==target else flag
                if column==target:
                    history_columns.append(key);history_sources[key]=column
    if horizon:
        # Features at t predict y[t+h]; purge the boundary before splitting later.
        frame[target] = frame[target].shift(-int(horizon))
    if repair and original_target is not None:
        frame.attrs['evaluation_target']=original_target.shift(-int(horizon)) if horizon else original_target
    frame.attrs['repair_values']=repaired_values
    frame.attrs['repair_records']=repair_records
    frame.attrs['repair_masks']=repair_masks
    frame.attrs['repair_originals']=repair_originals
    frame.attrs['causal_repair']=bool(repair)
    frame.attrs['history_columns']=history_columns
    frame.attrs['history_sources']=history_sources
    frame.attrs['persistence']=persistence
    frame.attrs.update(diagnostics)
    frame.attrs['frequency']=frequency
    frame.attrs['time'] = time
    frame.attrs['horizon'] = int(horizon)
    frame.attrs['lag_columns'] = [f'{source}_lag_{lag}' for source in sources for lag in lags]
    if time is not None:
        frame.index = pd.DatetimeIndex(time, name='datetime')
    return frame
