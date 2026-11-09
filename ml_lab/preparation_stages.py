"""Independent retrospective cleaning and reconstruction stages."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from ml_lab.repair import repair_signals
from ml_lab.model_repair import LearnedRepair


def staged_repair(frame, time, anomaly, restoration):
    output=frame.copy(); records=[]; masks={}; originals={}
    if anomaly:
        for column in anomaly.get('columns',[]):
            effective=anomaly; tuning={}
            if anomaly.get('auto_tune',False):
                from ml_lab.anomaly_tuning import tune_anomaly
                effective,tuning=tune_anomaly(frame[column],anomaly)
            output,rows,column_masks,column_originals=repair_signals(output,time,{**effective,'columns':[column],'fill_method':'none','anomaly_action':'replace'})
            for row in rows: row.update(tuning)
            records.extend(rows);masks.update(column_masks);originals.update(column_originals)
        for column in anomaly.get('columns',[]):
            original=originals[column]
            indexed=pd.Series(output[column].to_numpy(),index=pd.DatetimeIndex(time))
            treatment=anomaly.get('treatment','time')
            if treatment not in ('time','linear','polynomial','pchip','mask','median'):
                raise ValueError('Неизвестный способ обработки аномалий.')
            fallback=''
            if treatment=='mask':
                interpolated=indexed.to_numpy()
            elif treatment=='median':
                window=int(anomaly.get('window',48))
                interpolated=indexed.rolling(window,min_periods=1,center=True).median().to_numpy()
            else:
                try:
                    kwargs={'order':2} if treatment=='polynomial' else {}
                    interpolated=indexed.interpolate(method=treatment,limit_direction='both',**kwargs).ffill().bfill().to_numpy()
                except (ValueError,TypeError,ImportError) as exc:
                    fallback='Недостаточно опорных точек или неподходящие даты; использована временная интерполяция: '+str(exc)[:100]
                    interpolated=indexed.interpolate(method='time',limit_direction='both').to_numpy()
            # Stage one changes anomalous measurements only, leaving actual gaps to stage two.
            flags=masks[column]['anomaly']
            output.loc[flags,column]=interpolated[flags]
            records_column=next(row for row in records if row['Колонка']==column)
            records_column['Значений скорректировано']=int((flags & output[column].notna()).sum())
            records_column['Пропусков осталось']=int(output[column].isna().sum())
            masks[column]['corrected']=flags & output[column].notna().to_numpy()
            records_column['Этап']='Обработка аномалий'
            records_column['Способ обработки']=treatment
            records_column['Выбранная метка']=anomaly.get('anomaly_label',-1)
            records_column['Диагностика обработки']=fallback
    if restoration:
        if time is None: raise ValueError('Для восстановления нужен datetime.')
        stamps=pd.DatetimeIndex(time)
        seconds=(stamps-stamps[0]).total_seconds().to_numpy()/86400
        hour=stamps.hour+stamps.minute/60
        X=np.column_stack([seconds,np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                           np.sin(2*np.pi*stamps.dayofweek/7),np.cos(2*np.pi*stamps.dayofweek/7),
                           np.sin(2*np.pi*stamps.dayofyear/365.25),np.cos(2*np.pi*stamps.dayofyear/365.25)])
        for column in restoration.get('columns',[]):
            if column not in output: raise ValueError(f'Колонка восстановления отсутствует: {column}')
            values=pd.to_numeric(output[column],errors='coerce').replace([np.inf,-np.inf],np.nan)
            originals.setdefault(column,values.copy())
            missing=values.isna().to_numpy(); known=np.flatnonzero(~missing)
            if not len(known): raise ValueError(f'{column}: нет измерений для восстановления полностью пустой колонки.')
            config=restoration.get('model_config') or {}
            best=None;best_loss=None; chosen={}; failures=[]
            variants=[{}]
            name=config.get('model','ridge')
            if restoration.get('auto_tune',True):
                variants=([{'depth':depth} for depth in [3,6,10]] if name in ('random_forest','extra_trees','hist_gradient_boosting') else
                          [{'units':units} for units in [8,16,32]] if name in ('mlp','lstm','gru') else [{'alpha':a} for a in [.1,1.,10.]])
            if missing.any() and len(known)>=16:
                split=max(8,int(len(known)*.8));train=known[:split];valid=known[split:]
                for variant in variants:
                    try:
                        learner=LearnedRepair({**config,**variant});learner.positions=train.tolist()
                        model=learner.build()
                        if name=='ridge': model.steps[-1][1].regressor.set_params(alpha=variant.get('alpha',1.))
                        model.fit(X[train],values.iloc[train].to_numpy())
                        loss=mean_absolute_error(values.iloc[valid],model.predict(X[valid]))
                        if best_loss is None or loss<best_loss: best_loss=loss;chosen=variant
                    except (ValueError,RuntimeError,FloatingPointError) as exc: failures.append(str(exc)[:120])
                try:
                    learner=LearnedRepair({**config,**chosen});learner.positions=known.tolist();best=learner.build()
                    if name=='ridge': best.steps[-1][1].regressor.set_params(alpha=chosen.get('alpha',1.))
                    best.fit(X[known],values.iloc[known].to_numpy())
                    prediction=best.predict(X[np.flatnonzero(missing)])
                    values.iloc[np.flatnonzero(missing)]=np.where(np.isfinite(prediction),prediction,np.nan)
                except (ValueError,RuntimeError,FloatingPointError) as exc: failures.append(str(exc)[:120])
            model_count=int((missing & values.notna().to_numpy()).sum())
            values=values.interpolate(limit_direction='both').ffill().bfill()
            bound=(restoration.get('bounds') or {}).get(column,{})
            values=values.clip(lower=bound.get('min'),upper=bound.get('max'))
            output[column]=values
            masks.setdefault(column,dict(missing=missing,anomaly=np.zeros(len(frame),dtype=bool)))
            records.append({'Колонка':column,'Этап':'Восстановление пропусков','Модель восстановления':name,
                            'Пропусков до':int(missing.sum()),'Восстановлено моделью':model_count,
                            'Резервным способом':int(missing.sum())-model_count,'Пропусков осталось':int(values.isna().sum()),
                            'MAE временной валидации':best_loss,'Подобранные параметры':str(chosen),
                            'Предупреждение модели':'; '.join(failures)})
    return output,records,masks,originals
