"""Novelty detection on previous measured values, with batched out-of-sample scoring."""
import warnings
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from ml_lab.model_repair import integer

MODEL_DETECTORS=('isolation_forest','local_outlier_factor','one_class_svm','elliptic_envelope')
LABELS={'isolation_forest':'Isolation Forest','local_outlier_factor':'Local Outlier Factor',
        'one_class_svm':'One-Class SVM','elliptic_envelope':'Elliptic Envelope'}


def detector_settings(config):
    result={key:integer(config,key,default,low,high) for key,default,low,high in [
        ('min_samples',32,8,2048),('train_window',256,8,4096),('refit_interval',256,1,4096),
        ('max_fits',64,1,128),('trees',100,10,300),('neighbors',20,2,100),('seed',42,0,2147483647)]}
    if result['train_window']<result['min_samples']:
        raise ValueError('Окно детектора должно быть не меньше минимального числа измерений.')
    try:
        fraction=float(config.get('contamination',.05))
    except (ValueError,TypeError):
        raise ValueError('Доля аномалий должна быть числом от 0 до 0.5, не включая 0.') from None
    if not np.isfinite(fraction) or not 0<fraction<=.5:
        raise ValueError('Доля аномалий должна быть числом от 0 до 0.5, не включая 0.')
    result['contamination']=fraction
    return result


def build_detector(method, settings, count):
    fraction=settings['contamination']
    if method=='isolation_forest':
        estimator=IsolationForest(n_estimators=settings['trees'],contamination=fraction,random_state=settings['seed'],n_jobs=1)
    elif method=='local_outlier_factor':
        estimator=LocalOutlierFactor(n_neighbors=min(settings['neighbors'],count-1),contamination=fraction,novelty=True,n_jobs=1)
    elif method=='one_class_svm':
        estimator=OneClassSVM(nu=fraction,kernel='rbf',gamma='scale',max_iter=2000,cache_size=64)
    elif method=='elliptic_envelope':
        estimator=EllipticEnvelope(contamination=fraction,random_state=settings['seed'])
    else:
        raise ValueError('Неизвестный модельный детектор аномалий.')
    return make_pipeline(RobustScaler(),estimator)


def detect_model_anomalies(values, physical, method, config):
    """Fit at block starts; queries inside a block never enter that block's fit.

    All scoring methods are novelty detectors: queries are independent. Batching
    therefore reduces overhead without granting any query access to future rows.
    """
    if method not in MODEL_DETECTORS:
        raise ValueError('Неизвестный модельный детектор аномалий.')
    settings=detector_settings(config)
    observed=np.asarray(values,dtype=float)
    excluded=np.asarray(physical,dtype=bool)
    n=len(observed);flags=np.zeros(n,dtype=bool);support_outliers=np.zeros(n,dtype=bool);scores=np.full(n,np.nan)
    lower=np.full(n,np.nan);upper=np.full(n,np.nan)
    history=[];position=0;fits=attempts=constant_blocks=0;sample_count=guarded_count=0
    messages=set();model=None;reference=None;constant=False
    while position<n:
        if len(history)<settings['min_samples']:
            if np.isfinite(observed[position]) and not excluded[position]:
                history.append(float(observed[position]))
            position+=1
            continue
        if attempts<settings['max_fits']:
            reference=np.asarray(history[-settings['train_window']:],dtype=float)
            sample_count=len(reference);attempts+=1
            center=float(np.median(reference))
            floor=max(abs(center)*.01,1e-8)
            constant=float(np.ptp(reference))<=floor
            model=None
            if constant:
                constant_blocks+=1
            else:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model=build_detector(method,settings,len(reference))
                        model.fit(reference.reshape(-1,1))
                    fits+=1
                    messages.update(str(item.message)[:200] for item in caught)
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    model=None;messages.add('Резерв IQR: '+str(exc)[:180])
            end=min(n,position+settings['refit_interval'])
        else:
            # The fitted detector is frozen after the budget; score the rest in one batch.
            end=n
        center=float(np.median(reference));floor=max(abs(center)*.01,1e-8)
        valid=np.flatnonzero(np.isfinite(observed[position:end]) & ~excluded[position:end])+position
        lo,hi=np.quantile(reference,[settings['contamination']/2,1-settings['contamination']/2])
        lower[position:end]=lo;upper[position:end]=hi
        if len(valid):
            query=observed[valid]
            if constant:
                scores[valid]=np.abs(query-center)/(3*floor)-1
            elif model is not None:
                try:
                    scores[valid]=-np.asarray(model.decision_function(query.reshape(-1,1)),dtype=float)
                except (ValueError,RuntimeError,FloatingPointError) as exc:
                    messages.add('Резерв IQR при оценке: '+str(exc)[:150])
                    # Isolate failed queries: a later pathological value must not
                    # change an earlier row's scoring method inside this batch.
                    q1,q3=np.quantile(reference,[.25,.75]);spread=max(q3-q1,floor)
                    for index,value in zip(valid,query):
                        try:
                            scores[index]=-float(model.decision_function(np.asarray([[value]]))[0])
                        except (ValueError,RuntimeError,FloatingPointError):
                            scores[index]=max((q1-3*spread-value)/spread,(value-q3-3*spread)/spread)
            else:
                q1,q3=np.quantile(reference,[.25,.75]);spread=max(q3-q1,floor)
                scores[valid]=np.maximum((q1-3*spread-query)/spread,(query-q3-3*spread)/spread)
            # Tree detectors cannot distinguish how far an unseen value lies
            # beyond their last split. Retain a conservative past-IQR support guard.
            q1,q3=np.quantile(reference,[.25,.75]);spread=max(q3-q1,floor)
            guard=np.maximum((q1-3*spread-query)/spread,(query-q3-3*spread)/spread)
            guarded_count+=int(((guard>1e-9)&~(scores[valid]>1e-9)).sum())
            if not np.isfinite(scores[valid]).all():
                messages.add('Нечисловая оценка детектора: использована защита IQR')
            support_outliers[valid]=guard>1e-9
            scores[valid]=np.fmax(scores[valid],guard)
            flags[valid]=scores[valid]>1e-9
        accepted=observed[position:end][np.isfinite(observed[position:end]) & ~excluded[position:end] & ~support_outliers[position:end]]
        # Reconsider measured statistical candidates in new windows. Repeatedly
        # trimming each model's tails would shrink the normal support over time.
        # Imputed values never enter this pass, and extreme/physical spikes stay out.
        history.extend(accepted.tolist());history=history[-settings['train_window']:]
        position=end
    if constant_blocks:
        messages.add('Для почти постоянного прошлого окна применён порог отклонения 3% от уровня, минимум 3e-8')
    diagnostics={'Детектор аномалий':LABELS[method],'Обучений детектора':fits,'Дополнительных срабатываний IQR':guarded_count,
                 'Примеров при последнем обучении детектора':sample_count,
                 'Лимит обновлений детектора достигнут':attempts>=settings['max_fits'],
                 'Диагностика детектора':'; '.join(sorted(messages))}
    return flags,scores,lower,upper,diagnostics
