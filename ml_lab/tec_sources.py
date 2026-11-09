"""Validate and align measured/predicted TEC without fabricating timestamps or gap filling."""
import numpy as np
import pandas as pd
from ml_lab.radio_channels.tec import mapping, physical, C


def prepare_tec(frame, time_column, value_column, unit='TECU', kind='VTEC', elevation=60., height=350.):
    if time_column not in frame or value_column not in frame:
        raise ValueError('Выберите существующие колонки времени и ПЭС.')
    raw_time = frame[time_column]
    if pd.api.types.is_numeric_dtype(raw_time):
        raise ValueError('Числовое время неоднозначно. Используйте ISO datetime с часовым поясом или UTC.')
    from ml_lab.time_index import parse_datetime_values
    time = pd.to_datetime(raw_time,errors='coerce',utc=True) if pd.api.types.is_datetime64_any_dtype(raw_time) else parse_datetime_values(raw_time).dt.tz_localize('UTC')
    value = pd.to_numeric(frame[value_column], errors='coerce')
    valid = time.notna() & np.isfinite(value)
    clean = pd.DataFrame({'time': time[valid], 'tec': value[valid]}).sort_values('time')
    if clean.time.duplicated().any():
        raise ValueError('Обнаружены дубликаты времени. Устраните их в разделе «Данные».')
    if len(clean) < 2:
        raise ValueError('Нужно не менее двух корректных временных точек ПЭС.')
    if len(clean) > 50000:
        raise ValueError('Не более 50000 временных точек на опыт; выберите меньший участок.')
    if (clean.tec < 0).any():
        raise ValueError('ПЭС не может быть отрицательным; проверьте данные или прогноз.')
    if unit == 'electrons/m2':
        clean['tec'] /= 1e16
    elif unit != 'TECU':
        raise ValueError('Поддерживаются единицы TECU и electrons/m2.')
    clean['stec'] = clean.tec*(mapping(elevation, height) if kind == 'VTEC' else 1)
    if clean.stec.max()>2000:
        raise ValueError('STEC превышает 2000 TECU: проверьте единицы и выберите именно колонку ПЭС, а не другую радиохарактеристику.')
    times = clean.time.astype('int64').to_numpy()/1e9
    steps = np.diff(times)
    gap = steps > 3*np.median(steps)
    # Local adjacent differences only; never differentiate across large gaps.
    derivative = np.r_[np.nan, np.diff(clean.stec)/steps]
    derivative[np.r_[False, gap]] = np.nan
    clean['rate'] = derivative
    return clean.reset_index(drop=True), {'input_rows':len(frame), 'valid_rows':len(clean),
        'discarded_rows':int(len(frame)-len(clean)), 'gaps':int(gap.sum()), 'median_step_s':float(np.median(steps)),
        'timezone':'UTC', 'unit':'TECU', 'type':kind, 'mapping':float(mapping(elevation,height) if kind=='VTEC' else 1)}


def summarize_pair(measured, predicted, fc, bandwidth):
    joined = measured.merge(predicted, on='time', suffixes=('_real','_pred'), how='inner', validate='one_to_one')
    if len(joined) < 2:
        raise ValueError('У рядов нет двух общих временных меток UTC. Сдвиг и интерполяция автоматически не применяются.')
    seconds=joined.time.astype('int64').to_numpy()/1e9
    steps=np.diff(seconds)
    gaps=steps>3*np.median(steps)
    for name in ('real','pred'):
        rates=np.r_[np.nan,np.diff(joined[f'stec_{name}'])/steps]
        rates[np.r_[False,gaps]]=np.nan
        joined[f'rate_{name}']=rates
    metrics = {}
    for name, real, pred, unit in [
        ('stec', joined.stec_real, joined.stec_pred, 'TECU'),
        ('delay', physical(joined.stec_real,fc,bandwidth)['delay_ns'],physical(joined.stec_pred,fc,bandwidth)['delay_ns'],'ns'),
        ('phase', physical(joined.stec_real,fc,bandwidth)['phase_rad'],physical(joined.stec_pred,fc,bandwidth)['phase_rad'],'rad'),
        ('doppler',40.3e16*joined.rate_real/(C*fc),40.3e16*joined.rate_pred/(C*fc),'Hz')]:
        real, pred = np.asarray(real), np.asarray(pred)
        valid = np.isfinite(real)&np.isfinite(pred)
        error = pred[valid]-real[valid]
        metrics[name] = {'count':int(valid.sum()),'unit':unit,'mae':float(np.mean(np.abs(error))) if len(error) else None,
            'rmse':float(np.sqrt(np.mean(error**2))) if len(error) else None,
            'bias':float(np.mean(error)) if len(error) else None}
    truth = joined.stec_real.to_numpy(); pred = joined.stec_pred.to_numpy()
    nonzero = truth != 0
    metrics['stec']['mape_percent'] = float(100*np.mean(np.abs((pred[nonzero]-truth[nonzero])/truth[nonzero]))) if nonzero.any() else None
    denominator = float(np.sum((truth-truth.mean())**2))
    metrics['stec']['r2'] = float(1-np.sum((pred-truth)**2)/denominator) if denominator else None
    return joined, metrics


def serialize_series(frame, fc, bandwidth):
    data = physical(frame.stec.to_numpy(),fc,bandwidth)
    data['tec'] = frame.tec.to_numpy()
    data['rate_tecu_s'] = frame.rate.to_numpy()
    data['doppler_hz'] = 40.3e16*frame.rate.to_numpy()/(C*fc)
    data['time'] = frame.time.dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ').tolist()
    return {key: [float(v) if np.isfinite(v) else None for v in values] if key!='time' else values for key,values in data.items()}
