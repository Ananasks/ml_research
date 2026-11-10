"""Training, model generation and validation/test separation."""
import time
import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from ml_lab.neural import RecurrentRegressor

from ml_lab.model_registry import MODELS, MODEL_REGISTRY, RECURRENT_MODELS, build_estimator, sample_parameters

DEFAULTS = dict(seed=42, test_size=.2, validation_size=.2, split='time', trees=100, depth=None,
                min_samples_leaf=1, iterations=200, learning_rate=.05, alpha=.1,
                hidden_layers=[64, 32], activation='relu', early_stopping=False,
                C=1., kernel='rbf', epsilon=.1, neighbors=5, weights='uniform',
                units=32, layers=1, dropout=.1, dense_units=16, window=12, epochs=30,
                batch_size=64, optimizer='Adam', fit_intercept=True)


def make_pipeline(X, task, model_name, params):
    limit=MODEL_REGISTRY[model_name].get('max_rows')
    if limit and len(X)>limit:
        raise ValueError(f'{model_name}: максимум {limit} строк train из-за квадратичных затрат памяти. Уменьшите обучающую выборку или выберите другую модель.')
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    # Limit category expansion for identifier-like columns in user uploads.
    if any(X[c].nunique() > 200 for c in cat):
        raise ValueError('Категориальная колонка содержит более 200 значений. Исключите идентификаторы/дату или создайте календарные признаки.')
    scalers={'standard':StandardScaler(),'robust':RobustScaler(),'minmax':MinMaxScaler(),'none':'passthrough'}
    numeric = Pipeline([('imputer', SimpleImputer(strategy=params.get('imputer_strategy','median'), keep_empty_features=True,add_indicator=params.get('missing_indicators',False))), ('scale', scalers[params.get('scaler','standard')])])
    categorical = Pipeline([('imputer', SimpleImputer(strategy=params.get('categorical_imputer_strategy','most_frequent'), keep_empty_features=True)),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('num', numeric, num), ('cat', categorical, cat)], sparse_threshold=0)
    model = build_estimator(model_name, task, params)
    return Pipeline([('preprocess', prep), ('model', model)])


def evaluate(y, pred, task):
    y,pred=np.asarray(y),np.asarray(pred)
    if task == 'Классификация':
        valid=pd.notna(y)&pd.notna(pred)
        if not valid.any(): return {'Accuracy':float('nan'),'F1 macro':float('nan')}
        y,pred=y[valid].astype(str),pred[valid].astype(str)
        return {'Accuracy': float(accuracy_score(y, pred)), 'F1 macro': float(f1_score(y, pred, average='macro', zero_division=0))}
    y=pd.to_numeric(pd.Series(y),errors='coerce').to_numpy(dtype=float)
    pred=pred.astype(float)
    valid=np.isfinite(y)&np.isfinite(pred)
    y,pred=y[valid],pred[valid]
    if not len(y): return {key:float('nan') for key in ['R²','MAE','RMSE','MAPE, %']}
    nonzero = np.abs(y) > 1e-8
    return {'R²': float(r2_score(y, pred)) if len(y) > 1 else float('nan'), 'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'MAPE, %': float(np.mean(np.abs((y[nonzero]-pred[nonzero])/y[nonzero]))*100) if nonzero.any() else float('nan')}


def eligible_rows(df, features, target, task):
    if not features or target in features:
        raise ValueError('Выберите хотя бы один признак, исключая целевую колонку.')
    work = df[features + [target]].replace([np.inf, -np.inf], np.nan).reset_index(drop=True).copy()
    if task == 'Регрессия':
        work[target] = pd.to_numeric(work[target], errors='coerce')
    valid = work[target].notna()
    for c in df.attrs.get('history_columns', df.attrs.get('lag_columns', [])):
        if c in features:
            valid &= work[c].notna()
    work = work.loc[valid]
    work.attrs = {}
    return work


def split_data(df, features, target, task, params):
    p = DEFAULTS | params
    if df.attrs.get('causal_repair') and p['split']!='time':
        raise ValueError('Восстановление по прошлым наблюдениям требует разделения по времени.')
    work = eligible_rows(df,features,target,task)
    if len(work) < 30:
        raise ValueError('Нужно минимум 30 строк с известной целью и доступными лагами.')
    X, y = work[features], work[target]
    if task == 'Классификация':
        y = y.astype(str)
        if y.nunique() < 2:
            raise ValueError('Для классификации нужны минимум два класса.')
    positions = np.arange(len(work))
    if p.get('size_mode') == 'days':
        if p['split'] != 'time' or df.attrs.get('time') is None:
            raise ValueError('Разбиение по дням требует временного индекса и разделения по времени.')
        times = df.attrs['time'].reset_index(drop=True)
        gap = int(df.attrs.get('horizon', 0))
        source_times = times.iloc[work.index].reset_index(drop=True)
        target_times = times.shift(-gap).iloc[work.index].reset_index(drop=True)
        days = target_times.dt.normalize()
        valid = target_times.notna()
        if p.get('day_start'):
            valid &= days >= pd.Timestamp(p['day_start'])
        if p.get('day_end'):
            valid &= days <= pd.Timestamp(p['day_end'])
        if p.get('day_start') and p.get('day_end') and pd.Timestamp(p['day_start']) > pd.Timestamp(p['day_end']):
            raise ValueError('Начало диапазона дней должно быть не позже конца.')
        available = sorted(days[valid].unique())
        counts = [p.get('validation_days', 3), p.get('test_days', 1)]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or int(v) != v or v < 1 for v in counts):
            raise ValueError('Число дней валидации и теста должно быть положительным целым.')
        nval, ntest = map(int, counts)
        if len(available) <= nval + ntest:
            raise ValueError(f'В диапазоне {len(available)} доступных дней; нужны дни обучения, {nval} валидации и {ntest} теста.')
        validation_start, test_start = available[-nval-ntest], available[-ntest]
        train_idx = positions[(valid & (days < validation_start)).to_numpy()]
        val_idx = positions[(valid & (days >= validation_start) & (days < test_start)).to_numpy()]
        test_idx = positions[(valid & (days >= test_start)).to_numpy()]
        # Purge targets reaching the next partition's feature time, including horizon overlap.
        val_idx = val_idx[(target_times.iloc[val_idx] < source_times.iloc[test_idx[0]]).to_numpy()]
        if len(val_idx):
            train_idx = train_idx[(target_times.iloc[train_idx] < source_times.iloc[val_idx[0]]).to_numpy()]
    elif p.get('size_mode') == 'rows':
        sizes = [p.get(k) for k in ('train_rows','validation_rows','test_rows')]
        if any(v is None or int(v)!=v for v in sizes):
            raise ValueError('Укажите целое число строк для каждой части.')
        ntrain,nval,ntest=map(int,sizes)
        if ntrain<12 or nval<2 or ntest<1:
            raise ValueError('Минимум: 12 строк обучения, 2 валидации и 1 теста.')
        gap=int(df.attrs.get('horizon',0))
        times=df.attrs.get('time')
        source_times=times.iloc[work.index].reset_index(drop=True) if times is not None else None
        target_times=times.shift(-gap).iloc[work.index].reset_index(drop=True) if times is not None else None
        if p['split']=='time':
            if p.get('forecast_start'):
                if times is None: raise ValueError('Для периода прогноза нужна дата.')
                start,end=pd.Timestamp(p['forecast_start']),pd.Timestamp(p['forecast_end'])
                test_idx=positions[((target_times>=start)&(target_times<=end)).to_numpy()]
                if len(test_idx)!=ntest:
                    raise ValueError(f'В выбранном периоде {len(test_idx)} доступных тестовых строк, указано {ntest}.')
                if not len(test_idx): raise ValueError('В выбранном периоде нет тестовых строк.')
                before=positions[(target_times<start).to_numpy()]
                before=before[before<test_idx[0]-gap]
                required=ntrain+nval+gap
                if len(before)<required:
                    raise ValueError(f'Перед тестом доступно {len(before)} строк; для обучения, валидации и зазора нужно {required}.')
                block=before[-required:]
                train_idx,val_idx=block[:ntrain],block[ntrain+gap:]
            else:
                required=ntrain+nval+ntest+2*gap
                if required>len(work):
                    raise ValueError(f'Доступно {len(work)} строк; запрошено {required}, включая {2*gap} строк зазоров.')
                block=positions[-required:] if p.get('row_window')=='last' else positions[:required]
                train_idx=block[:ntrain]
                val_idx=block[ntrain+gap:ntrain+gap+nval]
                test_idx=block[ntrain+nval+2*gap:]
            if source_times is not None:
                for earlier,later in [(train_idx,val_idx),(val_idx,test_idx)]:
                    if target_times.iloc[earlier[-1]]>=source_times.iloc[later[0]]:
                        raise ValueError('Граница выборок пересекает одинаковые временные метки или горизонт прогноза. Измените число строк.')
        else:
            if gap or df.attrs.get('history_columns') or df.attrs.get('lag_columns'):
                raise ValueError('Лаги и горизонт требуют разделения по времени.')
            total=ntrain+nval+ntest
            if total>len(work): raise ValueError(f'Запрошено {total} строк, доступно {len(work)}.')
            try:
                chosen=positions
                if total<len(work):
                    chosen,_=train_test_split(positions,train_size=total,random_state=p['seed'],stratify=y if task=='Классификация' else None)
                rest,test_idx=train_test_split(chosen,test_size=ntest,random_state=p['seed'],stratify=y.iloc[chosen] if task=='Классификация' else None)
                train_idx,val_idx=train_test_split(rest,test_size=nval,random_state=p['seed'],stratify=y.iloc[rest] if task=='Классификация' else None)
            except ValueError as exc:
                raise ValueError('Указанные размеры не позволяют сохранить классы во всех выборках. Увеличьте число строк.') from exc
    elif p['split'] == 'time':
        times = df.attrs.get('time')
        gap = int(df.attrs.get('horizon', 0))
        target_times = times.shift(-gap).loc[work.index] if times is not None else None
        if p.get('forecast_start'):
            if target_times is None:
                raise ValueError('Для выбора даты прогноза нужна колонка datetime.')
            start, end = pd.Timestamp(p['forecast_start']), pd.Timestamp(p['forecast_end'])
            test_idx = positions[((target_times >= start) & (target_times <= end)).to_numpy()]
            preceding = positions[(target_times < start).to_numpy()]
            if not len(test_idx):
                raise ValueError('В выбранном интервале нет строк с известной целью. Для будущих дат используйте прогноз после обучения.')
            # Purge feature rows whose target reaches the forecast boundary.
            preceding = preceding[preceding < test_idx[0]-gap]
            test_start = int(test_idx[0])
            val_start = int(len(preceding)*(1-p['validation_size']/(1-p['test_size'])))
            if target_times is not None:
                while val_start > 0 and val_start < len(preceding) and target_times.iloc[preceding[val_start-1]] == target_times.iloc[preceding[val_start]]: val_start -= 1
            train_idx = preceding[:max(0,val_start-gap)]
            val_idx = preceding[val_start:]
        else:
            test_start = int(len(work)*(1-p['test_size']))
            val_start = int(len(work)*(1-p['test_size']-p['validation_size']))
            if target_times is not None:
                ordered_times = target_times.to_numpy()
                while test_start > 0 and ordered_times[test_start-1] == ordered_times[test_start]: test_start -= 1
                while val_start > 0 and ordered_times[val_start-1] == ordered_times[val_start]: val_start -= 1
            train_idx, val_idx, test_idx = positions[:max(0,val_start-gap)], positions[val_start:max(val_start,test_start-gap)], positions[test_start:]
    else:
        if df.attrs.get('history_columns') or df.attrs.get('lag_columns') or df.attrs.get('horizon', 0):
            raise ValueError('Лаги и горизонт прогноза требуют разделения по времени.')
        stratify = y if task == 'Классификация' else None
        try:
            rest, test_idx = train_test_split(positions, test_size=p['test_size'], random_state=p['seed'], stratify=stratify)
            train_idx, val_idx = train_test_split(rest, test_size=p['validation_size']/(1-p['test_size']), random_state=p['seed'],
                                                 stratify=y.iloc[rest] if stratify is not None else None)
        except ValueError as exc:
            raise ValueError('Недостаточно строк каждого класса для train/validation/test. Увеличьте датасет или измените доли.') from exc
    if len(train_idx) < 12 or len(val_idx) < 2 or len(test_idx) < 1:
        raise ValueError('После разделения осталось слишком мало строк. Уменьшите горизонт или измените доли.')
    if task == 'Классификация' and y.iloc[train_idx].nunique() < 2:
        raise ValueError('В обучающей части меньше двух классов.')
    # Object columns must have homogeneous strings for the categorical imputer/encoder.
    X = X.copy()
    for c in X.select_dtypes(exclude=np.number):
        X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    source_positions = work.index.to_numpy()
    if isinstance(df.index, pd.DatetimeIndex):
        X.index = df.index.take(source_positions).rename('datetime')
        y.index = X.index
    X.attrs = {}
    y.attrs = {}
    X.attrs['source_positions'] = source_positions.tolist()
    return X, y, (train_idx, val_idx, test_idx)


def generate_candidates(task, families, count, params, spaces=None):
    if not families:
        return []
    rng = np.random.default_rng(params.get('seed',42))
    result,seen=[],set()
    for attempt in range(count*50):
        if len(result)>=count: break
        name=families[attempt%len(families)]
        schema=spaces[name] if spaces is not None else {k:v for k,v in MODEL_REGISTRY[name]['parameters'].items() if v['type']!='layers'}
        q=sample_parameters(name,task,params,schema,rng=rng)
        key=repr((name,sorted(q.items())))
        if key not in seen:
            seen.add(key); result.append({'name':name,'params':q})
    return result


def row_positions(X):
    return np.asarray(X.attrs.get('source_positions', np.arange(len(X))), dtype=int)


def _context(pipeline, X, indices):
    model=pipeline.named_steps['model']
    first=int(indices[0])
    if isinstance(model,RecurrentRegressor):
        labels=row_positions(X)[indices]
        first_break=first==0 or labels[0]!=row_positions(X)[first-1]+1
        model.prediction_breaks_=np.r_[first_break,np.diff(labels)!=1]
    if isinstance(model,RecurrentRegressor) and model.window>1:
        preceding=X.iloc[max(0,first-model.window+1):first]
        gaps=np.flatnonzero(np.diff(row_positions(X)[max(0,first-model.window+1):first])!=1)
        if len(gaps): preceding=preceding.iloc[gaps[-1]+1:]
        context=np.asarray(pipeline.named_steps['preprocess'].transform(preceding),dtype=np.float32)
        if len(context)<model.window-1:
            context=np.concatenate([np.repeat(context[:1],model.window-1-len(context),axis=0),context])
        model.context_=context


def daily_metrics(predictions,task):
    if 'Время' not in predictions:
        return pd.DataFrame()
    rows=[]
    for day,group in predictions.groupby(pd.to_datetime(predictions['Время']).dt.date):
        rows.append({'День':pd.Timestamp(day),'Строк':len(group),'Измерений':int(group['Факт'].notna().sum()),**evaluate(group['Факт'],group['Прогноз'],task)})
    return pd.DataFrame(rows)


def run_experiments(df, features, target, task, candidates, params, progress=None, search=None):
    X,y,(train_idx,val_idx,test_idx)=split_data(df,features,target,task,params)
    actual=y.copy()
    if df.attrs.get('evaluation_target') is not None:
        actual=pd.Series(df.attrs['evaluation_target'].iloc[row_positions(X)].to_numpy(),index=y.index)
        if task=='Регрессия': actual=pd.to_numeric(actual,errors='coerce')
    metric='R²' if task=='Регрессия' else 'F1 macro'
    rows,best,best_score,validation_predictions=[],None,-float('inf'),[]
    study=None
    count=search['count'] if search else len(candidates)
    if search:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=params.get('seed',42),n_startup_trials=min(3,max(1,count//3))))
    times=df.attrs.get('time')
    target_times=times.shift(-int(df.attrs.get('horizon',0))) if times is not None else None
    for i in range(count):
        trial=None
        if study:
            trial=study.ask()
            name=trial.suggest_categorical('Модель',search['families'])
            p=sample_parameters(name,task,params,search['spaces'][name],trial=trial)
        else:
            name=candidates[i]['name']; p=params|candidates[i]['params']
        started=time.perf_counter()
        row={'Вариант':i+1,'Модель':name,'Параметры':p}
        try:
            if name in RECURRENT_MODELS and params.get('split','time')!='time':
                raise ValueError('Рекуррентная сеть требует разделения по времени.')
            pipeline=make_pipeline(X.iloc[train_idx],task,name,p)
            if isinstance(pipeline.named_steps['model'],RecurrentRegressor):
                pipeline.named_steps['model'].fit_breaks_=np.r_[True,np.diff(row_positions(X)[train_idx])!=1]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                pipeline.fit(X.iloc[train_idx],y.iloc[train_idx])
            _context(pipeline,X,val_idx)
            pred=pipeline.predict(X.iloc[val_idx])
            metrics=evaluate(actual.iloc[val_idx],pred,task)
            row['Измерений validation']=int(actual.iloc[val_idx].notna().sum())
            row.update(metrics,Статус='Готово',Предупреждения='\n'.join(dict.fromkeys(str(w.message) for w in caught)))
            comparison=pd.DataFrame({'Факт':actual.iloc[val_idx].to_numpy(),'Прогноз':pred,'Модель':f'{i+1} · {name}'},index=y.iloc[val_idx].index)
            if df.attrs.get('evaluation_target') is not None:
                comparison['Цель после восстановления']=y.iloc[val_idx].to_numpy()
                comparison['Факт измерен']=actual.iloc[val_idx].notna().to_numpy()
            if target_times is not None: comparison.insert(0,'Время',target_times.iloc[row_positions(X)[val_idx]].to_numpy())
            comparison.insert(0,'Индекс',row_positions(X)[val_idx])
            validation_predictions.append(comparison)
            if not np.isfinite(metrics[metric]): raise ValueError('Валидационная метрика не определена.')
            if study: study.tell(trial,metrics[metric])
            if metrics[metric]>best_score:
                best_score=metrics[metric]
                best={'pipeline':pipeline,'name':name,'params':p,'variant':i+1,'validation':metrics}
        except Exception as exc:
            row['Статус']=str(exc)
            if study and trial:
                import optuna
                study.tell(trial,state=optuna.trial.TrialState.FAIL)
        row['Секунды']=round(time.perf_counter()-started,2)
        rows.append(row)
        if progress: progress(i+1,count,name)
    if best is None: return {'leaderboard':rows,'error':'Ни одна модель не обучилась. Проверьте сообщения в таблице.'}
    _context(best['pipeline'],X,test_idx)
    pred=best['pipeline'].predict(X.iloc[test_idx])
    best['metrics']=evaluate(actual.iloc[test_idx],pred,task)
    if task=='Классификация' and hasattr(best['pipeline'], 'predict_proba'):
        best['test_probabilities']=best['pipeline'].predict_proba(X.iloc[test_idx])
        best['probability_classes']=best['pipeline'].classes_.astype(str).tolist()
    predictions=pd.DataFrame({'Факт':actual.iloc[test_idx].to_numpy(),'Прогноз':pred},index=y.iloc[test_idx].index)
    if target_times is not None: predictions.insert(0,'Время',target_times.iloc[row_positions(X)[test_idx]].to_numpy())
    predictions.insert(0,'Индекс',row_positions(X)[test_idx])
    if df.attrs.get('evaluation_target') is not None:
        predictions['Цель после восстановления']=y.iloc[test_idx].to_numpy()
        predictions['Факт измерен']=actual.iloc[test_idx].notna().to_numpy()
    persistence=df.attrs.get('persistence')
    if task=='Регрессия' and params.get('split','time')=='time' and persistence is not None:
        base=persistence.iloc[row_positions(X)[test_idx]].to_numpy(dtype=float)
        predictions['Базовый прогноз']=base
        mask=np.isfinite(base)&pd.to_numeric(predictions['Факт'],errors='coerce').notna().to_numpy()
        if mask.any():
            best['baseline']={'rows':int(mask.sum()),'metrics':evaluate(predictions['Факт'].to_numpy()[mask],base[mask],task),
                              'model_metrics':evaluate(predictions['Факт'].to_numpy()[mask],predictions['Прогноз'].to_numpy()[mask],task)}
    best.update(leaderboard=rows,predictions=predictions,features=features,target=target,task=task,
        comparison=pd.concat(validation_predictions),sizes={'train':len(train_idx),'validation':len(val_idx),'test':len(test_idx)},
        train_end_index=int(row_positions(X)[train_idx[-1]]),test_start_index=int(row_positions(X)[test_idx[0]]),
        explain_train=X.iloc[train_idx].head(40).copy(),explain_X=X.iloc[val_idx].head(100).copy(),explain_y=actual.iloc[val_idx].head(100).copy(),
        explain_context=X.iloc[max(0,int(val_idx[0])-96):int(val_idx[0])].copy(),
        explain_breaks=np.r_[val_idx[0]==0 or row_positions(X)[val_idx[0]]!=row_positions(X)[val_idx[0]-1]+1,np.diff(row_positions(X)[val_idx])!=1])
    # In-sample predictions diagnose fit; a private recurrent model keeps the
    # exported test context unchanged and reproduces the training window boundaries.
    fitted=copy.deepcopy(best['pipeline'])
    if isinstance(fitted.named_steps['model'],RecurrentRegressor):
        model=fitted.named_steps['model']
        first=np.asarray(fitted.named_steps['preprocess'].transform(X.iloc[train_idx[:1]]),dtype=np.float32)
        model.context_=np.repeat(first,max(0,model.window-1),axis=0)
        model.prediction_breaks_=np.r_[True,np.diff(row_positions(X)[train_idx])!=1]
    train_pred=fitted.predict(X.iloc[train_idx])
    training=pd.DataFrame({'Факт':actual.iloc[train_idx].to_numpy(),'Прогноз':train_pred},index=X.iloc[train_idx].index)
    if df.attrs.get('evaluation_target') is not None:
        training['Цель после восстановления']=y.iloc[train_idx].to_numpy()
        training['Факт измерен']=actual.iloc[train_idx].notna().to_numpy()
    training.insert(0,'Индекс',row_positions(X)[train_idx])
    if target_times is not None:
        training.insert(0,'Время',target_times.iloc[row_positions(X)[train_idx]].to_numpy())
        training.index=pd.DatetimeIndex(training['Время'],name='datetime')
    if task=='Регрессия' and params.get('split','time')=='time' and persistence is not None:
        training['Базовый прогноз']=persistence.iloc[row_positions(X)[train_idx]].to_numpy(dtype=float)
        best['baseline_label']='Предыдущее доступное значение цели (восстановленное при необходимости)'
    else:
        constant=float(y.iloc[train_idx].median()) if task=='Регрессия' else y.iloc[train_idx].mode().iloc[0]
        training['Базовый прогноз']=constant
        best['predictions']['Базовый прогноз']=constant
        best['baseline_label']='Медиана цели train' if task=='Регрессия' else 'Наиболее частый класс train'
    best['train_predictions']=training
    best['train_metrics']=evaluate(training['Факт'],training['Прогноз'],task)
    best['evaluation_sizes']={'validation':int(actual.iloc[val_idx].notna().sum()),'test':int(actual.iloc[test_idx].notna().sum()),
        'restored_train':int((actual.iloc[train_idx].isna()&y.iloc[train_idx].notna()).sum())}
    best['explain_context_positions']=row_positions(X)[max(0,int(val_idx[0])-96):int(val_idx[0])]
    if target_times is not None:
        best['predictions'].index=pd.DatetimeIndex(best['predictions']['Время'],name='datetime')
        best['comparison'].index=pd.DatetimeIndex(best['comparison']['Время'],name='datetime')
    return best


def train(df, features, target, task, model_name, params):
    result=run_experiments(df,features,target,task,[{'name':model_name,'params':params}],params)
    if 'error' in result: raise ValueError(result['leaderboard'][0]['Статус'])
    return result['pipeline'],result['metrics'],result['predictions']
