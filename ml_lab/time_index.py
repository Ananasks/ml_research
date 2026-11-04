"""Canonical datetime preparation. Real timestamps remain a DatetimeIndex only."""
import pandas as pd

def detect_time_columns(df):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    if 'datetime' in lookup:
        return [lookup['datetime']]
    if 'date' in lookup and 'time' in lookup:
        return [lookup['date'], lookup['time']]
    if 'date' in lookup:
        return [lookup['date']]
    return []


def parse_datetime_values(text,dayfirst=None,strict=True):
    """Parse one consistent date convention per column, never invent a date for a clock.

    Research CSVs use ISO datetime (MUF/Athens) or DD-MM-YYYY + time (TEC/CB).
    ISO year-first strings are never interpreted with day/month reversal.
    """
    text=text.astype(str).str.strip()
    iso=text.str.match(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:\D|$)')
    full_date=text.str.contains(r'(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})',regex=True)
    clock_only=text.str.fullmatch(r'\d{1,3}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?')
    valid=(full_date if strict else ~clock_only) & ~clock_only
    if dayfirst is None:
        parts=text.str.extract(r'^(\d{1,2})([-/.])(\d{1,2})[-/.]\d{4}')
        first=pd.to_numeric(parts[0],errors='coerce')
        second=pd.to_numeric(parts[2],errors='coerce')
        if first.gt(12).any():
            dayfirst=True
        elif second.gt(12).any():
            dayfirst=False
        else:
            # Day-first is the documented convention of the research DD-MM-YYYY files.
            dayfirst=bool(parts[1].isin(['-','.']).any())
    parsed=pd.to_datetime(text.where(valid & ~iso),errors='coerce',dayfirst=bool(dayfirst),format='mixed',utc=True)
    parsed.loc[iso]=pd.to_datetime(text.where(valid & iso).loc[iso],errors='coerce',format='mixed',utc=True)
    return parsed.dt.tz_localize(None)



def normalize_time_index(df,time_columns=None,dayfirst=None,strict_dates=True,drop_invalid_time=False,
                         frequency=None,duplicate_policy='error',start_datetime=None,index_frequency=None,target=None):
    detected=detect_time_columns(df)
    if detected and str(detected[0]).strip().lower()=='datetime': time_columns=detected
    elif not time_columns: time_columns=detected
    frame=df.copy();frame.attrs={}
    source='none'
    if time_columns:
        if target in time_columns: raise ValueError('Целевая колонка не может быть колонкой времени.')
        text=frame[list(time_columns)].astype(str).agg(' '.join,axis=1).reset_index(drop=True)
        time=parse_datetime_values(text,dayfirst,strict_dates)
        source='datetime' if str(time_columns[0]).strip().lower()=='datetime' else 'date + time' if len(time_columns)>1 else 'date'
    elif isinstance(df.index,pd.DatetimeIndex):
        time=pd.Series(pd.to_datetime(df.index,utc=True).tz_localize(None))
        source='DatetimeIndex'
    elif start_datetime is not None:
        if not index_frequency: raise ValueError('Для создания индекса укажите известный шаг измерений.')
        start=parse_datetime_values(pd.Series([start_datetime]),strict=True).iloc[0]
        if pd.isna(start): raise ValueError('Укажите полную начальную дату индекса.')
        if pd.Timedelta(index_frequency)<=pd.Timedelta(0): raise ValueError('Шаг индекса должен быть положительным.')
        time=pd.Series(pd.date_range(start,periods=len(frame),freq=index_frequency))
        source='generated'
    else:
        frame=frame.reset_index(drop=True)
        return frame,None,dict(invalid_time_rows=0,grid_added_rows=0,duplicates_removed=0,time_source=source)
    frame=frame.reset_index(drop=True)
    bad=time.isna()
    invalid=int(bad.sum())
    if invalid:
        if not drop_invalid_time: raise ValueError(f'Не удалось прочитать время в {invalid} строках. Проверьте формат даты.')
        frame=frame.loc[~bad].reset_index(drop=True);time=time.loc[~bad].reset_index(drop=True)
        if frame.empty: raise ValueError('Не найдено ни одной корректной даты.')
    order=time.sort_values(kind='stable').index
    frame=frame.loc[order].reset_index(drop=True);time=time.loc[order].reset_index(drop=True)
    # A timestamp is an index, never an estimator input column.
    frame=frame.drop(columns=[c for c in frame if str(c).strip().lower()=='datetime' or c in time_columns])
    added=removed=0
    if frequency:
        delta=pd.Timedelta(frequency)
        if delta<=pd.Timedelta(0): raise ValueError('Шаг сетки должен быть положительным.')
        duplicated=time.duplicated(keep='last')
        if duplicated.any():
            if duplicate_policy!='last': raise ValueError('Для сетки нужны уникальные даты. Выберите обработку дубликатов.')
            removed=int(duplicated.sum());frame=frame.loc[~duplicated].reset_index(drop=True);time=time.loc[~duplicated].reset_index(drop=True)
        start,end=time.min(),time.max()
        if (end-start)/delta>1_000_000: raise ValueError('Сетка слишком велика. Увеличьте шаг.')
        grid=pd.date_range(start,end,freq=delta)
        if not time.isin(grid).all(): raise ValueError('Некоторые даты не совпадают с выбранной сеткой. Округление измерений не выполняется.')
        added=len(grid)-len(frame)
        frame.index=pd.DatetimeIndex(time)
        frame=frame.reindex(grid).reset_index(drop=True);time=pd.Series(grid)
    return frame,time,dict(invalid_time_rows=invalid,grid_added_rows=added,duplicates_removed=removed,time_source=source)
