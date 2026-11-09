"""Bounded retrospective parameter search using explicit synthetic proxy labels."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from ml_lab.anomaly_detection import MODEL_DETECTORS, detect_model_anomalies, detector_settings


def tune_anomaly(values, config):
    """Never claim proxy quality is accuracy on genuine, unlabelled anomalies."""
    method=config.get('anomaly_method','isolation_forest')
    measured=pd.to_numeric(values,errors='coerce').to_numpy(dtype=float)
    measured=measured[np.isfinite(measured)]
    bound=(config.get('bounds') or {}).get(str(values.name),{})
    if bound.get('min') is not None: measured=measured[measured>=float(bound['min'])]
    if bound.get('max') is not None: measured=measured[measured<=float(bound['max'])]
    diagnostic={'Автоподбор детектора':'Не выполнен','Проверено конфигураций':0}
    if method not in (*MODEL_DETECTORS,'iqr','mad','zscore'):
        return config,{**diagnostic,'Причина автоподбора':'Метод не поддерживает автоподбор.'}
    if len(measured)<96:
        return config,{**diagnostic,'Причина автоподбора':'Нужно минимум 96 конечных измерений в физических границах. Сохранены ручные настройки.'}
    settings=detector_settings(config.get('anomaly_config') or {}) if method in MODEL_DETECTORS else {}
    minimum=settings.get('min_samples',8)
    windows=[n for n in (64,128,256) if n>=minimum and n+32<=len(measured)]
    if not windows:
        return config,{**diagnostic,'Причина автоподбора':'Недостаточно наблюдений для заданной минимальной истории. Сохранены ручные настройки.'}
    candidates=[(window,threshold) for window in windows for threshold in ((.01,.05,.1) if method in MODEL_DETECTORS else (2.,3.,4.))]
    best=None;attempts=0;failures=0
    for window,threshold in candidates:
        labels=[];predictions=[];false_positives=[]
        try:
            # Two time-separated reference/query blocks; no query enters its fit.
            for end in sorted(set([max(window+32,len(measured)//2),len(measured)])):
                block=measured[max(0,end-window-64):end]
                reference=block[:window];clean=block[window:]
                if len(clean)<16: continue
                q1,q3=np.quantile(reference,[.25,.75]);center=np.median(reference)
                scale=max(q3-q1,abs(center)*.01,1e-8)
                # Keep plausible held-out observations as proxy negatives, not verified labels.
                clean=np.clip(clean,q1-1.5*scale,q3+1.5*scale)
                corrupt=clean.copy();injected=np.arange(0,len(clean),8)
                corrupt[injected]=center+np.where(np.arange(len(injected))%2,1,-1)*scale*np.resize([3.,5.,8.],len(injected))
                query=np.r_[clean,corrupt];expected=np.zeros(len(query),dtype=bool);expected[len(clean)+injected]=True
                if method in MODEL_DETECTORS:
                    params={**settings,'train_window':window,'min_samples':len(reference),'contamination':threshold,'max_fits':1,'refit_interval':4096}
                    flags,*_=detect_model_anomalies(np.r_[reference,query],np.zeros(len(reference)+len(query),dtype=bool),method,params)
                    predicted=flags[len(reference):]
                else:
                    if method=='iqr': low,high=q1-threshold*scale,q3+threshold*scale
                    elif method=='mad':
                        spread=max(np.median(np.abs(reference-center))*1.4826,abs(center)*.01,1e-8)
                        low,high=center-threshold*spread,center+threshold*spread
                    else:
                        center=np.mean(reference);spread=max(np.std(reference),abs(center)*.01,1e-8)
                        low,high=center-threshold*spread,center+threshold*spread
                    predicted=(query<low)|(query>high)
                labels.extend(expected);predictions.extend(predicted)
                false_positives.extend(predicted[~expected])
            if not labels: continue
            attempts+=1
            f1=float(f1_score(labels,predictions,zero_division=0));fpr=float(np.mean(false_positives))
            # Precision/F1 discourages trivial marking of the complete signal.
            score=f1-.5*fpr
            if best is None or score>best[0]: best=(score,window,threshold,f1,fpr)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError): failures+=1
    if best is None:
        return config,{**diagnostic,'Причина автоподбора':'Проверка конфигураций не удалась. Сохранены ручные настройки.'}
    _,window,threshold,f1,fpr=best
    if method in MODEL_DETECTORS:
        selected={'train_window':window,'contamination':threshold}
        updated={**config,'anomaly_config':{**(config.get('anomaly_config') or {}),**selected}}
    else:
        selected={'window':window,'threshold':threshold}
        updated={**config,**selected,'min_history':min(int(config.get('min_history',8)),window)}
    return updated,{'Автоподбор детектора':'По синтетическим выбросам','Параметры детектора':json.dumps(selected,ensure_ascii=False),
                    'Проверено конфигураций':attempts,'F1 синтетической проверки':round(f1,4),
                    'Доля ложных срабатываний на контрольных точках':round(fpr,4),'Ошибок конфигураций':failures,
                    'Причина автоподбора':'Прокси-проверка на искусственных одиночных всплесках. Качество на реальных аномалиях неизвестно; проверьте график.'}
