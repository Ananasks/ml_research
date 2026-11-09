import numpy as np
import pandas as pd
from ml_lab.anomaly_tuning import tune_anomaly
from ml_lab.data import prepare_frame


def signal(n=180):
    return pd.Series(10+np.sin(np.arange(n)/9),name='y')


def test_bounded_deterministic_search_preserves_manual_configuration():
    config={'anomaly_method':'isolation_forest','anomaly_label':1,'anomaly_config':{'trees':10,'seed':42}}
    first,diagnostics=tune_anomaly(signal(),config)
    second,again=tune_anomaly(signal(),config)
    assert first==second and diagnostics==again
    assert 'train_window' not in config['anomaly_config']
    assert first['anomaly_label']==1
    assert 0<diagnostics['Проверено конфигураций']<=9
    assert 0<=diagnostics['F1 синтетической проверки']<=1


def test_insufficient_and_invalid_values_keep_manual_settings():
    config={'anomaly_method':'iqr','threshold':5}
    for values in (signal(10),pd.Series([np.nan]*100,name='y'),pd.Series([np.inf]*100,name='y')):
        chosen,diagnostics=tune_anomaly(values,config)
        assert chosen==config
        assert diagnostics['Проверено конфигураций']==0
        assert 'Причина автоподбора' in diagnostics


def test_statistical_thresholds_and_stage_integration():
    raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=180,freq='h'),'y':signal()})
    raw.loc[120,'y']=200
    prepared=prepare_frame(raw,target='y',anomaly_stage={'columns':['y'],'anomaly_method':'mad','auto_tune':True})
    record=prepared.attrs['repair_records'][0]
    assert prepared.y.iloc[120]<20
    assert record['Автоподбор детектора']=='По синтетическим выбросам'
    assert record['Проверено конфигураций']>0
    assert 'threshold' in record['Параметры детектора']
