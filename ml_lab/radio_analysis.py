"""Uniform-segment channel diagnostics: Durbin-Levinson PACF, ApEn and PSD."""
import numpy as np
import pandas as pd
from scipy.signal import correlate, welch, periodogram, detrend, find_peaks
from scipy.spatial import cKDTree


def approximate_entropy(values, m=2, ratio=.2):
    x=np.asarray(values,dtype=float)
    if len(x)<m+2 or not np.isfinite(x).all(): raise ValueError('Для ApEn нужны конечные значения и N ≥ m + 2.')
    deviation=float(np.std(x,ddof=0))
    if deviation==0: return 0.,0.
    radius=ratio*deviation
    def phi(size):
        patterns=np.lib.stride_tricks.sliding_window_view(x,size).copy()
        counts=cKDTree(patterns).query_ball_point(patterns,radius,p=np.inf,return_length=True)
        # Self-matches are deliberately included: this is ApEn, not sample entropy.
        return float(np.mean(np.log(counts/len(patterns))))
    return phi(m)-phi(m+1),radius


def partial_correlation(x,lags):
    centered=np.asarray(x,dtype=float)-np.mean(x)
    scale=np.max(np.abs(centered))
    if scale==0: return [1.]+[None]*lags
    centered=centered/scale
    cov=correlate(centered,centered,mode='full',method='fft')[len(x)-1:len(x)+lags]/len(x)
    variance=float(cov[0]);coeff=np.empty(0);result=[1.]
    for k in range(1,lags+1):
        if variance<=np.finfo(float).eps*cov[0]: return result+[None]*(lags+1-len(result))
        reflection=float((cov[k]-np.dot(coeff,cov[1:k][::-1]))/variance)
        if not np.isfinite(reflection) or abs(reflection)>1+1e-8: return result+[None]*(lags+1-len(result))
        result.append(reflection)
        coeff=np.r_[coeff-reflection*coeff[::-1],reflection]
        variance*=max(0.,1-reflection**2)
    return result


def analyze_channel(frame,column,max_points=4096,lags=48,m=2,ratio=.2,method='welch',segment=512,remove_trend='constant'):
    if column not in frame: raise ValueError('Выбранный столбец отсутствует.')
    if not isinstance(frame.index,pd.DatetimeIndex): raise ValueError('Для анализа задайте datetime в разделе «Данные».')
    if not frame.index.is_unique: raise ValueError('Для анализа нужны уникальные datetime. Настройте обработку дубликатов в разделе «Данные».')
    if not 128<=max_points<=8192 or not 1<=lags<=200 or not 1<=m<=4 or not .01<=ratio<=1:
        raise ValueError('Проверьте длину участка, число лагов, m и r.')
    if method not in ('welch','periodogram') or remove_trend not in ('constant','linear') or not 16<=segment<=8192:
        raise ValueError('Проверьте настройки спектрального анализа.')
    data=pd.to_numeric(frame[column],errors='coerce').sort_index().replace([np.inf,-np.inf],np.nan)
    deltas=data.index.to_series().diff().dropna()
    positive=deltas[deltas>pd.Timedelta(0)]
    if positive.empty: raise ValueError('Недостаточно временных отсчётов.')
    step=positive.mode().iloc[0];seconds=step.total_seconds()
    finite=data.notna().to_numpy()
    breaks=np.r_[True,np.diff(data.index.asi8)!=step.value] | ~finite | np.r_[False,~finite[:-1]]
    groups=np.cumsum(breaks)
    pieces=[part for _,part in data[finite].groupby(groups[finite])]
    if not pieces: raise ValueError('В выбранном столбце нет числовых измерений.')
    longest=max(pieces,key=len);selected=longest.iloc[-max_points:]
    if len(selected)<32: raise ValueError('Нужно минимум 32 последовательных отсчёта без пропусков. Восстановите ряд или выберите другой участок.')
    x=selected.to_numpy(dtype=float);n=len(x)
    adjusted=detrend(x,type=remove_trend)
    # Normalize PACF/ApEn inputs to avoid scale-dependent numerical precision for CB.
    normalized=(adjusted-np.mean(adjusted))
    sigma=float(np.std(normalized))
    constant=float(np.ptp(x))==0 or sigma<=10*np.finfo(float).eps*max(1.,float(np.max(np.abs(x))))
    if constant: normalized=np.zeros(n)
    elif sigma>0: normalized=normalized/sigma
    count=min(lags,n//2-1)
    pacf=partial_correlation(normalized,count)
    bound=1.96/np.sqrt(n)
    entropy,radius=approximate_entropy(normalized,m,ratio)
    entropy_windows=[]
    width=min(512,n)
    for start in range(0,n-width+1,width):
        value,_=approximate_entropy(normalized[start:start+width],m,ratio)
        entropy_windows.append({'time':selected.index[start+width//2].isoformat(),'value':value,'count':width})
    used_segment=min(segment,n)
    estimator=welch if method=='welch' else periodogram
    kwargs={'nperseg':used_segment,'noverlap':used_segment//2} if method=='welch' else {}
    frequency,power=estimator(x,fs=1/seconds,window='hann',detrend=remove_trend,scaling='density',**kwargs)
    if constant: power=np.zeros_like(power)
    peaks=find_peaks(power)[0]
    if len(power)>2 and power[-1]>power[-2]: peaks=np.r_[peaks,len(power)-1]
    peaks=peaks[(frequency[peaks]>0)&(power[peaks]>0)]
    if not len(peaks) and not constant and np.any(power[1:]>0): peaks=np.asarray([1+np.argmax(power[1:])])
    ranked=sorted(peaks,key=lambda i:power[i],reverse=True)[:5]
    peak_rows=[{'frequencyHz':float(frequency[i]),'cyclesPerDay':float(frequency[i]*86400),
                'periodHours':float(1/frequency[i]/3600),'density':float(power[i]),
                'cyclesInSegment':float(frequency[i]*(n-1)*seconds)} for i in ranked]
    notes=[]
    if n<len(data): notes.append('Анализ ограничен непрерывным участком. Пропуски не удаляются со склеиванием времени.')
    if len(longest)>max_points: notes.append(f'Взят последний участок из {max_points} отсчётов самого длинного непрерывного фрагмента.')
    if count<lags: notes.append(f'Число лагов ограничено до {count} из-за длины участка.')
    if constant: notes.append('После удаления среднего/тренда ряд постоянный в пределах численной точности: ApEn = 0; PACF не определена, спектральных пиков нет.')
    return {'column':column,'count':n,'totalRows':len(data),'finiteRows':int(finite.sum()),'longestSegment':len(longest),
            'start':selected.index[0].isoformat(),'end':selected.index[-1].isoformat(),'stepSeconds':seconds,
            'durationHours':(n-1)*seconds/3600,'trend':remove_trend,'notes':notes,
            'series':{'time':[t.isoformat() for t in selected.index],'values':x.tolist()},
            'pacf':{'lags':list(range(count+1)),'values':pacf,'lagMinutes':[i*seconds/60 for i in range(count+1)],'bound95':float(bound)},
            'apen':{'value':float(entropy),'m':m,'ratio':ratio,'radiusNormalized':float(radius),'radiusOriginalUnits':float(radius*sigma),'windows':entropy_windows},
            'spectrum':{'method':method,'segment':used_segment if method=='welch' else n,
                        'frequencyHz':frequency[1:].tolist(),'cyclesPerDay':(frequency[1:]*86400).tolist(),
                        'density':power[1:].tolist(),'peaks':peak_rows,'nyquistHz':.5/seconds}}
