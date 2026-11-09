import numpy as np
import pandas as pd
import pytest
from ml_lab.data import prepare_frame
from ml_lab.diagnostics import daily_metric_figures


@pytest.mark.parametrize('step',['5min','15min','30min'])
def test_restoration_enforces_grid_and_drops_saved_row_index(step):
    raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=140,freq=step),
                      'index':np.arange(140),'Unnamed: 0':np.arange(140),'x':np.sin(np.arange(140)/8)+10})
    raw=raw.drop(index=[20,21,60,61])
    frame=prepare_frame(raw,target='x',restoration={'columns':['x'],'auto_tune':False,'model_config':{'model':'ridge'}})
    assert isinstance(frame.index,pd.DatetimeIndex)
    assert frame.index.name=='datetime'
    assert len(frame)==140
    assert (frame.index.to_series().diff().dropna()==pd.Timedelta(step)).all()
    assert not {'datetime','index','Unnamed: 0'}.intersection(frame.columns)
    assert frame.x.notna().all()
    assert frame.attrs['frequency']==str(pd.Timedelta(step))


def test_prepared_daily_metrics_fill_real_data_gaps_without_fabricating_observations():
    time=pd.date_range('2024-01-01',periods=144,freq='30min')
    actual=np.repeat([10.,np.nan,12.],48)
    prepared=np.repeat([10.,11.,12.],48)
    frame=pd.DataFrame({'Время':time,'Факт':actual,'Цель после восстановления':prepared,'Прогноз':prepared+.5},index=time)
    result={'task':'Регрессия','predictions':frame,'train_predictions':frame,'comparison':frame}
    observed=daily_metric_figures(result)
    complete=daily_metric_figures(result,basis='prepared')
    measured_y=np.asarray(observed['daily:test:MAE'].data[0].y,dtype=float)
    prepared_y=np.asarray(complete['daily_prepared:test:MAE'].data[0].y,dtype=float)
    assert np.isnan(measured_y[1])
    np.testing.assert_allclose(prepared_y,.5)
    assert frame['Факт'].isna().sum()==48
