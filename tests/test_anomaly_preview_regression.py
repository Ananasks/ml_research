from concurrent.futures import ThreadPoolExecutor
from threading import Event
import pandas as pd
import api.main as api


def test_unrelated_preparation_does_not_wait_for_slow_restore(monkeypatch):
    started=Event();release=Event()
    raw=pd.DataFrame({'x':[1.,2.],'y':[3.,4.]})
    calls=[]
    def prepare(frame,**options):
        calls.append(options['restoration']['columns'][0])
        if options['restoration']['columns']==['x']:
            started.set()
            assert release.wait(5)
        return frame.copy()
    monkeypatch.setattr(api,'prepare_frame',prepare)
    monkeypatch.setattr(api,'_PREPARED_MODELS',api.OrderedDict())
    monkeypatch.setattr(api,'_PREPARING_MODELS',{})
    with ThreadPoolExecutor(max_workers=3) as pool:
        first=pool.submit(api.prepare_dataset,raw,'y','Регрессия',{'restoration':{'columns':['x']}})
        assert started.wait(2)
        duplicate=pool.submit(api.prepare_dataset,raw,'y','Регрессия',{'restoration':{'columns':['x']}})
        independent=pool.submit(api.prepare_dataset,raw,'y','Регрессия',{'restoration':{'columns':['y']}})
        try:
            assert len(independent.result(timeout=2)[0])==2
        finally:
            release.set()
        first.result(timeout=2);duplicate.result(timeout=2)
    assert calls.count('x')==1
    assert calls.count('y')==1


def test_cb_lof_auto_preview_returns_finite_series():
    result=api.preview_preparation('tec_cb.csv',api.PreparationRequest(target='tetu_absCB',preparation={
        'drop_invalid_time':True,'calendar':False,
        'anomaly_stage':{'columns':['tetu_absCB'],'anomaly_method':'local_outlier_factor','auto_tune':True}}))
    points=result['repairSeries']['tetu_absCB']
    assert len(points)>100
    assert sum(point['before'] is not None for point in points)>100
    assert any(point['anomaly'] for point in points)
    assert 'index' not in result['features']
    assert all(isinstance(point['x'],str) for point in points)


def test_old_cb_draft_with_service_index_still_builds_anomaly_graph():
    result=api.preview_preparation('tec_cb.csv',api.PreparationRequest(target='tetu_absCB',preparation={
        'drop_invalid_time':True,'calendar':False,
        'anomaly_stage':{'columns':['index','tetu_absCB'],'anomaly_method':'local_outlier_factor','auto_tune':True},
        'feature_columns':['index','tetu_absCB']}))
    assert 'tetu_absCB' in result['repairSeries']
    assert 'index' not in result['repairSeries']
    assert 'index' not in result['features']
