import numpy as np
import pandas as pd
import pytest
from ml_lab.data import prepare_frame


def data():
    t=pd.date_range('2024-01-01',periods=90,freq='30min')
    return pd.DataFrame({'datetime':t,'y':10+np.sin(np.arange(90)/5),'x':np.arange(90,dtype=float)})


def test_independent_anomaly_stage_leaves_real_gaps():
    raw=data();raw.loc[40,'y']=500;raw.loc[60,'y']=np.nan
    result=prepare_frame(raw,target='y',anomaly_stage={'columns':['y'],'anomaly_method':'iqr'})
    assert result.y.iloc[40]<20
    assert np.isnan(result.y.iloc[60])
    assert result.attrs['repair_records'][0]['Пропусков осталось']==1
    assert result.attrs['repair_masks']['y']['anomaly'][40]


def test_restores_grid_initial_and_internal_gaps_and_tunes():
    raw=data().drop(index=[30,31]);raw.loc[[0,1,40],'y']=np.nan
    result=prepare_frame(raw,target='y',frequency='auto',restoration={'columns':['y'],'auto_tune':True,'model_config':{'model':'ridge'}},calendar_features=['calendar_hour'],lags=[1],feature_columns=['y'])
    assert len(result)==90
    assert result.y.notna().all()
    assert result.index[1]-result.index[0]==pd.Timedelta('30min')
    assert 'y_lag_1' in result
    assert 'calendar_hour' in result and 'calendar_month' not in result
    assert result.attrs['repair_records'][0]['MAE временной валидации'] is not None
    assert result.attrs['evaluation_target'].isna().sum()==5


def test_empty_column_is_reported():
    raw=data();raw['y']=np.nan
    with pytest.raises(ValueError,match='нет измерений'):
        prepare_frame(raw,target='y',restoration={'columns':['y']})


@pytest.mark.parametrize('model',['ridge','random_forest','extra_trees','hist_gradient_boosting','mlp','lstm','gru'])
def test_models_restore_selected_gaps(model):
    raw=data();raw.loc[50:52,'y']=np.nan
    result=prepare_frame(raw,target='y',restoration={'columns':['y'],'auto_tune':False,'model_config':{'model':model,'epochs':1,'iterations':20}})
    record=result.attrs['repair_records'][0]
    assert result.y.notna().all()
    assert record['Восстановлено моделью']==3,record


def test_reverse_label_selects_scored_normal_points_only():
    from ml_lab.repair import repair_signals
    raw=data();raw.loc[40,'y']=500;raw.loc[60,'y']=np.nan
    times=raw.datetime
    config={'columns':['y'],'anomaly_method':'iqr','fill_method':'none','anomaly_action':'replace'}
    _,_,negative,_=repair_signals(raw,times,config)
    _,_,positive,_=repair_signals(raw,times,{**config,'anomaly_label':1})
    assert negative['y']['anomaly'][40]
    assert not positive['y']['anomaly'][40]
    assert not positive['y']['anomaly'][:8].any()
    assert not positive['y']['anomaly'][60]
    eligible=raw.y.shift(1).rolling(48,min_periods=8).count().ge(8) & raw.y.notna()
    assert np.array_equal(positive['y']['anomaly']|negative['y']['anomaly'],eligible.to_numpy())


@pytest.mark.parametrize('treatment',['time','linear','polynomial','pchip','median','mask'])
def test_anomaly_treatments_preserve_rows_and_original_gaps(treatment):
    raw=data();raw.loc[40,'y']=500;raw.loc[60,'y']=np.nan
    result=prepare_frame(raw,target='y',anomaly_stage={'columns':['y'],'anomaly_method':'iqr','treatment':treatment})
    assert len(result)==len(raw)
    assert np.isnan(result.y.iloc[60])
    if treatment=='mask': assert np.isnan(result.y.iloc[40])
    else: assert np.isfinite(result.y.iloc[40]) and result.y.iloc[40]<20
    assert result.attrs['repair_records'][0]['Способ обработки']==treatment


def test_model_label_reverse_excludes_warmup_and_retains_physical_violations():
    from ml_lab.repair import repair_signals
    raw=data();raw.loc[70,'y']=500;raw.loc[60,'y']=np.nan
    config={'columns':['y'],'anomaly_method':'isolation_forest','anomaly_label':1,
            'anomaly_config':{'scaler':'standard'},'bounds':{'y':{'max':100}},'fill_method':'none'}
    _,_,masks,_=repair_signals(raw,raw.datetime,config)
    assert not masks['y']['anomaly'][:32].any()
    assert not masks['y']['anomaly'][60]
    assert masks['y']['anomaly'][70]
