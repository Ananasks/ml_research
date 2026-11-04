"""Causal signal repair: no future rows or global statistics enter past values."""
import numpy as np
import pandas as pd
from ml_lab.anomaly_detection import MODEL_DETECTORS, detector_settings, detect_model_anomalies


def repair_signals(frame, times, config):
    output=frame.copy()
    method=config.get('anomaly_method','iqr')
    fill=config.get('fill_method','linear')
    action=config.get('anomaly_action','clip')
    window=int(config.get('window',48)); minimum=int(config.get('min_history',8))
    threshold=float(config.get('threshold',3.))
    max_gap=int(config.get('max_gap',0))
    learned_config=config.get('model_config') or {}
    if fill=='model':
        from ml_lab.model_repair import LearnedRepair
        LearnedRepair(learned_config)  # Validate even if no columns or gaps are selected.
    if method in MODEL_DETECTORS: detector_settings(config.get('anomaly_config') or {})
    if method not in ('none','iqr','mad','zscore',*MODEL_DETECTORS) or fill not in ('none','forward','linear','model') or action not in ('clip','replace'):
        raise ValueError('Неизвестный метод восстановления или поиска аномалий.')
    if not 2<=minimum<=window or window>10000 or threshold<=0 or max_gap<0:
        raise ValueError('Проверьте окно, минимальную историю, порог и длину пропуска.')
    if times is None: raise ValueError('Восстановлению временного ряда нужен datetime.')
    # Seconds relative to the first timestamp keep the regression slope well conditioned.
    seconds=(pd.DatetimeIndex(times)-pd.Timestamp(times.iloc[0])).total_seconds().to_numpy()
    records=[]; masks={}; originals={}
    for column in config.get('columns',[]):
        if column not in frame: raise ValueError(f'Колонка восстановления отсутствует: {column}')
        values=pd.to_numeric(frame[column],errors='coerce').replace([np.inf,-np.inf],np.nan)
        originals[column]=values.copy()
        bounds=config.get('bounds',{}).get(column,{})
        low=bounds.get('min'); high=bounds.get('max')
        if low is not None and high is not None and low>high: raise ValueError(f'{column}: минимум больше максимума.')
        physical=pd.Series(False,index=values.index)
        if low is not None: physical |= values.lt(float(low))
        if high is not None: physical |= values.gt(float(high))
        reference=values.mask(physical).shift(1).rolling(window,min_periods=minimum)
        center=reference.median()
        floor=np.maximum(center.abs()*float(config.get('relative_floor',.01)),1e-8)
        if method=='iqr':
            q1=reference.quantile(.25); q3=reference.quantile(.75)
            spread=(q3-q1).clip(lower=floor)
            lower=q1-threshold*spread; upper=q3+threshold*spread
        elif method=='mad':
            spread=reference.apply(lambda x:float(np.nanmedian(np.abs(x-np.nanmedian(x)))),raw=True)*1.4826
            spread=spread.clip(lower=floor)
            lower=center-threshold*spread; upper=center+threshold*spread
        elif method=='zscore':
            center=reference.mean(); spread=reference.std(ddof=0).clip(lower=floor)
            lower=center-threshold*spread; upper=center+threshold*spread
        else:
            lower=pd.Series(np.nan,index=values.index); upper=lower.copy()
        statistical=(values.lt(lower)|values.gt(upper)) & values.notna()
        detector_diagnostics={};anomaly_score=None
        if method in MODEL_DETECTORS:
            detected,anomaly_score,model_lower,model_upper,detector_diagnostics=detect_model_anomalies(
                values,physical,method,config.get('anomaly_config') or {})
            statistical=pd.Series(detected,index=values.index)
            lower=pd.Series(model_lower,index=values.index);upper=pd.Series(model_upper,index=values.index)
        anomalies=physical|statistical
        cleaned=values.copy()
        if action=='clip':
            cleaned=cleaned.where(~statistical,cleaned.clip(lower=lower,upper=upper))
            if low is not None: cleaned=cleaned.clip(lower=float(low))
            if high is not None: cleaned=cleaned.clip(upper=float(high))
        else: cleaned=cleaned.mask(anomalies)
        # Only accepted observations anchor extrapolation; reconstructed rows do not
        # recursively become new evidence and cannot create accelerating drift.
        result=cleaned.to_numpy(dtype=float,copy=True)
        learner=LearnedRepair(learned_config) if fill=='model' else None
        last=[]; gap=0
        for i in range(len(result)):
            vector=learner.features(pd.Timestamp(times.iloc[i]),seconds[i]) if learner else None
            if np.isfinite(result[i]):
                if learner:
                    if not anomalies.iloc[i]:
                        learner.accept(vector,float(result[i]),seconds[i],i)
                    learner.remember(vector)
                last.append(i); last=last[-2:]; gap=0
                continue
            gap+=1
            if learner:
                estimate=learner.predict(vector) if not max_gap or gap<=max_gap else None
                learner.remember(vector)
            else:
                estimate=None
            if fill=='none' or not last or (max_gap and gap>max_gap): continue
            used_model=estimate is not None
            reserve=learner.fallback if learner else fill
            if estimate is None:
                if reserve=='none': continue
                estimate=result[last[-1]]
            if not used_model and reserve=='linear' and len(last)==2:
                before,after=last
                elapsed=seconds[after]-seconds[before]
                if elapsed>0:
                    estimate+=(result[after]-result[before])*(seconds[i]-seconds[after])/elapsed
            if low is not None: estimate=max(float(low),estimate)
            if high is not None: estimate=min(float(high),estimate)
            if np.isfinite(estimate):
                result[i]=estimate
                if learner:
                    if used_model: learner.model_filled+=1
                    else: learner.fallback_filled+=1
        output[column]=result
        restored=values.isna() & np.isfinite(result)
        corrected=values.notna() & np.isfinite(result) & ~np.isclose(values.to_numpy(),result,rtol=0,atol=0,equal_nan=True)
        masks[column]=dict(missing=values.isna().to_numpy(),anomaly=anomalies.to_numpy(),
            restored=restored.to_numpy(),corrected=corrected.to_numpy())
        records.append({'Колонка':column,'Пропусков до':int(values.isna().sum()),'Аномалий':int(anomalies.sum()),
            'Пропусков восстановлено':int(restored.sum()),'Значений скорректировано':int(corrected.sum()),
            'Пропусков осталось':int(pd.isna(result).sum())})
        if anomaly_score is not None: masks[column]['anomaly_score']=anomaly_score
        records[-1].update(detector_diagnostics)
        if learner: records[-1].update(learner.diagnostics())
    return output,records,masks,originals
