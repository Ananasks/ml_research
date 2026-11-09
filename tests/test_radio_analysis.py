import numpy as np
import pandas as pd
import pytest
from ml_lab.radio_analysis import analyze_channel,approximate_entropy,partial_correlation


def frame(values,step='15min'):
    return pd.DataFrame({'cb':values},index=pd.date_range('2024-01-01',periods=len(values),freq=step,name='datetime'))


def test_pacf_matches_ar1_coefficient_and_excludes_intermediate_effect():
    rng=np.random.default_rng(42);x=np.zeros(4000)
    for i in range(1,len(x)): x[i]=.7*x[i-1]+rng.normal()
    values=partial_correlation(x,8)
    assert values[1]==pytest.approx(.7,abs=.04)
    assert max(abs(value) for value in values[2:])<.06


def test_apen_constant_and_regular_vs_noise():
    rng=np.random.default_rng(1)
    assert approximate_entropy(np.ones(100))[0]==0
    regular=np.tile([0.,1.,0.,-1.],256)
    assert approximate_entropy(regular)[0]<approximate_entropy(rng.normal(size=1024))[0]


def test_entropy_uses_self_matches_and_chebyshev_distance():
    x=np.array([.1,.2,.4,.5,.3,.1,.2,.7])
    m=2;r=.2*np.std(x)
    def phi(length):
        patterns=np.lib.stride_tricks.sliding_window_view(x,length)
        counts=[np.mean(np.max(np.abs(patterns-v),axis=1)<=r) for v in patterns]
        return np.mean(np.log(counts))
    assert approximate_entropy(x,m,.2)[0]==pytest.approx(phi(m)-phi(m+1))


@pytest.mark.parametrize('method',['welch','periodogram'])
def test_spectrum_recovers_known_physical_period(method):
    n=2048;period=32
    result=analyze_channel(frame(np.sin(2*np.pi*np.arange(n)/period)), 'cb',method=method,segment=512)
    assert result['stepSeconds']==900
    assert result['spectrum']['peaks'][0]['periodHours']==pytest.approx(8.,rel=.04)
    assert result['count']==n
    assert result['pacf']['lagMinutes'][1]==15


def test_no_stitching_across_missing_values_or_missing_times():
    data=frame(np.sin(np.arange(300)/5))
    data.iloc[100:105]=np.nan
    data=data.drop(data.index[200:205])
    result=analyze_channel(data,'cb')
    assert result['count']==100
    assert result['end']==data.index[99].isoformat()
    assert result['notes']


def test_constant_signal_has_no_misleading_peaks_or_pacf():
    result=analyze_channel(frame(np.ones(256)*1e9),'cb')
    assert result['apen']['value']==0
    assert result['spectrum']['peaks']==[]
    assert all(value is None for value in result['pacf']['values'][1:])


def test_duplicate_or_short_data_are_rejected():
    data=frame(np.arange(64,dtype=float));data.index=pd.DatetimeIndex([data.index[0]]*64)
    with pytest.raises(ValueError,match='уникальные'): analyze_channel(data,'cb')
    with pytest.raises(ValueError,match='32'): analyze_channel(frame(np.ones(10)),'cb')


def test_api_source_and_no_inherited_feature_engineering(monkeypatch):
    import api.main as api
    raw=frame(np.sin(np.arange(200)/5)).reset_index()
    monkeypatch.setattr(api,'get_dataset',lambda name:raw)
    result=api.radio_analysis('sample.csv',api.RadioAnalysisRequest(column='cb',preparation={'lags':[900],'feature_columns':['missing'],'horizon':20}))
    assert result['count']==200
    assert result['source']=='prepared'
    assert not result['processingApplied']
