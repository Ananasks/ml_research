import pandas as pd
from api.tec import StationRequest, station_field


def test_station_field_preserves_utc_missing_and_station_values(monkeypatch):
    import api.main
    frame=pd.DataFrame({'date':['29-12-2022','29-12-2022'], 'time':['00:00:00','00:30:00'],
                        'a_absTEC':[5,float('nan')], 'b_absTEC':[7,8]})
    monkeypatch.setattr(api.main,'get_dataset',lambda _:frame)
    result=station_field(StationRequest(dataset='stations.csv',times=['2022-12-29T00:30:00Z']))
    assert result['stations']==['a','b']
    assert result['time']==['2022-12-29T00:30:00+00:00']
    assert result['values']==[[None],[8.0]]


def test_single_station_does_not_invent_spatial_field(monkeypatch):
    import api.main
    monkeypatch.setattr(api.main,'get_dataset',lambda _:pd.DataFrame({'datetime':['2022-12-29'], 'a_absTEC':[5]}))
    assert station_field(StationRequest(dataset='single.csv',times=[]))['stations']==[]
