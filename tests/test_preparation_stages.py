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
