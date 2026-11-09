import numpy as np
from sklearn.linear_model import Ridge
from ml_lab.run_store import RunStore


def test_full_fitted_run_survives_restart(tmp_path):
    model=Ridge().fit([[1],[2],[3]],[2,4,6])
    run={'pipeline':model,'config':{'dataset':'example.csv'},'predictions':np.array([2.,4.])}
    first=RunStore(tmp_path);first['abc123']=run
    restarted=RunStore(tmp_path)
    restored=restarted.get('abc123')
    assert restored['config']==run['config']
    np.testing.assert_allclose(restored['pipeline'].predict([[2]]),model.predict([[2]]))
    np.testing.assert_array_equal(restored['predictions'],run['predictions'])


def test_missing_corrupt_and_unsafe_identifiers(tmp_path):
    store=RunStore(tmp_path)
    assert store.get('missing') is None
    assert store.get('../../escape') is None
    (tmp_path/'broken.joblib').write_bytes(b'broken')
    assert store.get('broken') is None


def test_diagnostics_series_and_explanation_load_after_restart(tmp_path,monkeypatch):
    import pandas as pd
    import api.main as api
    from ml_lab.core import run_experiments
    from ml_lab.data import prepare_frame
    raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=120,freq='h'),'x':np.arange(120,dtype=float),'y':np.arange(120,dtype=float)*2+1})
    result=run_experiments(prepare_frame(raw,target='y'),['x'],'y','Регрессия',[{'name':'Ridge','params':{}}],{'split':'time'})
    RunStore(tmp_path).__setitem__('saved',result)
    monkeypatch.setattr(api,'RUNS',RunStore(tmp_path))
    figures=api.experiment_diagnostics('saved')
    assert 'daily:test:MAE' in figures
    assert api.experiment_series('saved')['totals']['test']==result['sizes']['test']
    explained=api.explain_experiment('saved',{'method':'Permutation Importance'})
    assert explained['payload']
    # Explanation is saved along with the run too.
    assert RunStore(tmp_path).get('saved').get('explanations')


def test_recurrent_pipeline_snapshot_keeps_prediction_context(tmp_path):
    import pandas as pd
    from ml_lab.core import run_experiments
    from ml_lab.data import prepare_frame
    raw=pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=80,freq='h'),'x':np.sin(np.arange(80)/8),'y':np.sin(np.arange(80)/8)})
    result=run_experiments(prepare_frame(raw,target='y'),['x'],'y','Регрессия',[{'name':'LSTM','params':{'epochs':1,'units':8,'window':3}}],{'split':'time'})
    store=RunStore(tmp_path);store['neural']=result
    restored=RunStore(tmp_path).get('neural')
    np.testing.assert_allclose(restored['pipeline'].predict(raw[['x']].iloc[-10:]),result['pipeline'].predict(raw[['x']].iloc[-10:]))
