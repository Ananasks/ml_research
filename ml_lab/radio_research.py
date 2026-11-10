"""Day-wise descriptive TEC analysis. Fits describe observed data, never train a forecast."""
from typing import Literal
from datetime import timezone
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import welch, correlate
from pydantic import BaseModel, Field
from ml_lab.radio_daily import DailySample
from ml_lab.radio_channels.tec import physical, C


class ResearchRequest(BaseModel):
    samples: list[DailySample] = Field(min_length=1, max_length=50000)
    method: Literal['harmonic', 'lowess', 'spline', 'none'] = 'harmonic'
    smoothing: float = Field(.3, ge=.05, le=1)
    carrier_mhz: float = Field(1575.42, ge=100, le=12000)
    bandwidth_mhz: float = Field(1, ge=.001, le=200)


def approximation(seconds, values, method, smoothing):
    x, y = np.asarray(seconds, float), np.asarray(values, float)
    if method == 'none':
        return y.copy()
    if len(y) < 3:
        return None if method == 'harmonic' else y.copy()
    if method == 'harmonic':
        design = np.column_stack([np.ones(len(x)), *[f(2*np.pi*x/p) for p in (86400,43200) for f in (np.cos,np.sin)]])
        if len(x) < design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
            return None
        return design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if method == 'spline':
        return UnivariateSpline(x-x[0], y, k=min(3,len(y)-1), s=len(y)*np.var(y)*smoothing)(x-x[0])
    if len(y) > 4096:
        raise ValueError('LOWESS: не более 4096 точек в сутки. Выберите другой метод или меньший период.')
    width = max(3, int(np.ceil(len(y)*smoothing)))
    fitted = []
    for t in x:
        distances = np.abs(x-t)
        radius = np.partition(distances, width-1)[width-1]
        weights = (1-np.minimum(distances/max(radius,1e-12),1)**3)**3
        design = np.column_stack([np.ones(len(x)), (x-t)/max(radius,1)])
        root = np.sqrt(weights)
        fitted.append(np.linalg.lstsq(design*root[:,None], y*root, rcond=None)[0][0])
    return np.asarray(fitted)


def statistics(y):
    y = np.asarray(y,float); y=y[np.isfinite(y)]
    if not len(y): return {'count':0,'mean':None,'variance':None,'std':None,'min':None,'max':None,'q25':None,'median':None,'q75':None}
    return {'count':len(y),'mean':float(y.mean()),'variance':float(y.var()),'std':float(y.std()),
            'min':float(y.min()),'max':float(y.max()),'q25':float(np.quantile(y,.25)),
            'median':float(np.median(y)),'q75':float(np.quantile(y,.75))}


def errors(real, predicted):
    a,b=np.asarray(real,float),np.asarray(predicted,float)
    mask=np.isfinite(a)&np.isfinite(b);a,b=a[mask],b[mask];d=b-a
    if not len(a): return {'count':0,'mae':None,'rmse':None,'bias':None,'max_abs':None,'r2':None,'mape_percent':None,'interval95':None,'std':None}
    total=np.sum((a-a.mean())**2)
    return {'count':len(a),'mae':float(np.mean(abs(d))),'rmse':float(np.sqrt(np.mean(d*d))),
            'bias':float(d.mean()),'max_abs':float(np.max(abs(d))),
            'r2':float(1-np.sum(d*d)/total) if len(a)>1 and total>0 else None,
            'mape_percent':float(100*np.mean(np.abs(d[np.abs(a)>1e-8]/a[np.abs(a)>1e-8]))) if np.any(np.abs(a)>1e-8) else None,
            'interval95':np.quantile(d,[.025,.975]).tolist(),'std':float(d.std())}


def temporal_diagnostics(t,y):
    # Never join gaps or different days. Use a genuinely uniform contiguous segment.
    x=np.asarray(t,float);y=np.asarray(y,float)
    if len(x)<8:return {'lags_s':[],'acf':[],'frequency_hz':[],'psd':[],'count':0,'note':'Для ACF/PSD нужно минимум 8 последовательных равномерных точек.'}
    steps=np.diff(x);step=float(np.median(steps))
    cuts=np.r_[0,np.flatnonzero(~np.isclose(steps,step,rtol=1e-5,atol=.001))+1,len(x)]
    start,stop=max(zip(cuts[:-1],cuts[1:]),key=lambda pair:pair[1]-pair[0])
    v=y[start:stop];n=len(v)
    if n<8:return {'lags_s':[],'acf':[],'frequency_hz':[],'psd':[],'count':n,'note':'Нет равномерного непрерывного участка из 8 точек; интерполяция не применяется.'}
    centered=v-v.mean();power=float(np.dot(centered,centered));lags=np.arange(min(n//2,49))
    acf=correlate(centered,centered,mode='full',method='fft')[n-1:n-1+len(lags)]/power if power>1e-20 else np.full(len(lags),np.nan)
    f,p=welch(v,fs=1/step,nperseg=min(256,n),detrend='constant')
    return {'lags_s':(lags*step).tolist(),'acf':[float(v) if np.isfinite(v) else None for v in acf],
            'frequency_hz':f[1:].tolist(),'psd':p[1:].tolist(),'count':n,
            'start_seconds_utc':float(x[start]),'note':f'Непрерывный участок: {n} точек; шаг {step:g} с. ACF с вычитанием среднего, Welch PSD; сутки не склеиваются.'}


def analyze_research(request: ResearchRequest):
    fc,bw=request.carrier_mhz*1e6,request.bandwidth_mhz*1e6
    if bw>=2*fc:raise ValueError('Полоса пересекает нулевую частоту.')
    samples=sorted([s.model_copy(update={'time':s.time.astimezone(timezone.utc)}) for s in request.samples],key=lambda s:s.time)
    if len({s.time for s in samples})!=len(samples):raise ValueError('Нужны уникальные временные метки.')
    if any(s.measured is None and s.predicted is None for s in samples):raise ValueError('В каждой точке нужен ПЭС.')
    sources={};daily_errors=[]
    for name in ('measured','predicted'):
        available=[s for s in samples if getattr(s,name) is not None]
        if not available:continue
        all_fits=[];daily=[];all_seconds=[];trends=[];seasonals=[];remainders=[]
        for day in sorted({s.time.date() for s in available}):
            group=[s for s in available if s.time.date()==day]
            seconds=np.array([(s.time-s.time.replace(hour=0,minute=0,second=0,microsecond=0)).total_seconds() for s in group])
            values=np.array([getattr(s,name) for s in group])
            fitted=approximation(seconds,values,request.method,request.smoothing)
            all_fits.extend(fitted.tolist() if fitted is not None else [None]*len(group));all_seconds.extend(seconds.tolist())
            design=np.column_stack([np.ones(len(seconds)),(seconds-43200)/86400,*[f(2*np.pi*seconds/p) for p in (86400,43200) for f in (np.cos,np.sin)]])
            if len(group)>=6 and np.linalg.matrix_rank(design)==6:
                coefficients=np.linalg.lstsq(design,values,rcond=None)[0]
                trend=design[:,:2]@coefficients[:2];seasonal=design[:,2:]@coefficients[2:]
                trends.extend(trend.tolist());seasonals.extend(seasonal.tolist());remainders.extend((values-trend-seasonal).tolist())
            else:
                trends.extend([None]*len(group));seasonals.extend([None]*len(group));remainders.extend([None]*len(group))
            daily.append({'day':day.isoformat(),**statistics(values),'diagnostics':temporal_diagnostics(seconds,values),
                          'fit_note':'Гармонической регрессии нужны минимум 5 независимых точек.' if fitted is None else ''})
        values=np.array([getattr(s,name) for s in available]);physical_values=physical(values,fc,bw)
        series={k:[float(x) if np.isfinite(x) else None for x in v] for k,v in physical_values.items()}
        rates=np.array([getattr(s,name+'_rate') for s in available],float)
        # Derivatives on the same common intervals, no rates supplied from unmatched intervals.
        for i,s in enumerate(available):
            if i==0 or s.time.date()!=available[i-1].time.date():rates[i]=np.nan
            else:
                dt=(s.time-available[i-1].time).total_seconds()
                day_steps=[(available[j].time-available[j-1].time).total_seconds() for j in range(max(1,i-48),i+1) if available[j].time.date()==s.time.date() and available[j-1].time.date()==s.time.date()]
                rates[i]=(values[i]-values[i-1])/dt if dt<=3*np.median(day_steps) else np.nan
        series.update({'time':[s.time.isoformat() for s in available],
            'rate_tecu_s':[float(r) if np.isfinite(r) else None for r in rates],
            'doppler_hz':[float(40.3e16*r/(C*fc)) if np.isfinite(r) else None for r in rates],
            'delay_frequency_derivative_s_per_hz':(-80.6e16*values/(C*fc**3)).tolist(),
            'dispersion_symbol_ratio':(np.array(physical_values['dispersion_ns'])*1e-9*bw).tolist()})
        sources[name]={'series':series,'fit':all_fits,'residual':[float(v-f) if f is not None else None for v,f in zip(values,all_fits)],
                       'statistics':statistics(values),'daily':daily,
                       'decomposition':{'trend':trends,'seasonal':seasonals,'residual':remainders,'method':'Joint least squares: linear trend + 24h/12h harmonics, separately per UTC day'},
                       'hourly_profile':[{'hour':float(hour/3600),**statistics(values[np.asarray(all_seconds)==hour])} for hour in sorted(set(all_seconds))]}
    for day in sorted({s.time.date().isoformat() for s in samples})+['all']:
        group=[s for s in samples if s.measured is not None and s.predicted is not None and (day=='all' or s.time.date().isoformat()==day)]
        if not group:continue
        a=np.array([s.measured for s in group]);b=np.array([s.predicted for s in group])
        row={'day':day,'stec':errors(a,b)}
        for field in ('delay_ns','phase_rad','path_m','dispersion_ns'):
            row[field]=errors(physical(a,fc,bw)[field],physical(b,fc,bw)[field])
        # Different missing-data masks must not compare derivatives over different intervals.
        real_rates=[];pred_rates=[]
        for date in sorted({s.time.date() for s in group}):
            points=[s for s in group if s.time.date()==date]
            dt=np.array([(points[i].time-points[i-1].time).total_seconds() for i in range(1,len(points))])
            if not len(dt):continue
            valid=dt<=3*np.median(dt)
            real_rates.extend((np.diff([s.measured for s in points])/dt)[valid].tolist())
            pred_rates.extend((np.diff([s.predicted for s in points])/dt)[valid].tolist())
        row['doppler_hz']=errors(40.3e16*np.asarray(real_rates)/(C*fc),40.3e16*np.asarray(pred_rates)/(C*fc))
        daily_errors.append(row)
    spread={key:statistics([row['stec'][key] for row in daily_errors if row['day']!='all' and row['stec'][key] is not None]) for key in ('mae','rmse','bias','max_abs')}
    return {'sources':sources,'errors':daily_errors,'day_spread':spread,'method':request.method,
            'notes':['Аппроксимация строится отдельно внутри каждого UTC-дня и описывает наблюдения; это не ML-прогноз.',
                     'Для измерений и прогноза используются одинаковые физические формулы. Кларк не создаёт ПЭС или ионосферную задержку.',
                     'dτ/df = −80.6 S/(c f³), с/Гц; это частотная производная, а не статистическая дисперсия.',
                     'ΔτB/Ts использует Ts=1/B — интервал отсчётов; для OFDM отношение к полезному символу равно (ΔτB/Ts)/Nfft.']}


class ResponseRequest(BaseModel):
    stec_tecu: float = Field(ge=0, le=2000)
    carrier_mhz: float = Field(1575.42, ge=100, le=12000)
    bandwidth_mhz: float = Field(1, ge=.001, le=200)


def ionosphere_response(request: ResponseRequest):
    fc,bw=request.carrier_mhz*1e6,request.bandwidth_mhz*1e6
    if bw>=2*fc:raise ValueError('Полоса пересекает нулевую частоту.')
    n=512;nu=np.fft.fftfreq(n,1/bw);k=2*np.pi*40.3e16*request.stec_tecu/C
    # Remove constant phase and common group delay: show only residual dispersion.
    response=np.exp(1j*k*(1/(fc+nu)-1/fc+nu/fc**2))
    t=(np.arange(n)-n//2)/bw
    pulse=np.exp(-.5*(t/(3/bw))**2).astype(complex)
    received=np.fft.ifft(np.fft.fft(pulse)*response)
    impulse=np.fft.fftshift(np.fft.ifft(response))
    def width(x):
        p=np.abs(x)**2;p=p/p.sum();center=float(np.dot(p,t))
        return float(np.sqrt(np.dot(p,(t-center)**2))*1e9)
    return {'time_ns':(t*1e9).tolist(),'impulse_amplitude':np.abs(impulse).tolist(),
            'tx_real':pulse.real.tolist(),'rx_real':received.real.tolist(),
            'tx_intensity':(np.abs(pulse)**2).tolist(),'rx_intensity':(np.abs(received)**2).tolist(),
            'tx_rms_width_ns':width(pulse),'rx_rms_width_ns':width(received),
            'note':'Полосоограниченный DFT-отклик: 512 отсчётов, Fs=B; постоянная фаза и общая групповая задержка удалены. Гауссов импульс σ=3/B; модель периодическая, окно ограничено. Поглощение и случайные замирания отсутствуют.'}
