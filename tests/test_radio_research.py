import numpy as np
import pytest
from ml_lab.radio_research import (ResearchRequest, ResponseRequest, analyze_research,
    approximation, temporal_diagnostics, ionosphere_response)
from ml_lab.clarke_statistics import fading_statistics
from ml_lab.radio_channels.tec import C, physical


def samples():
    return [{'time':f'2024-01-{day:02d}T{hour:02d}:00:00Z',
             'measured':20+5*np.cos(2*np.pi*hour/24),
             'predicted':21+5*np.cos(2*np.pi*hour/24)} for day in (1,2) for hour in range(24)]


@pytest.mark.parametrize('method',['harmonic','lowess','spline'])
def test_research_fits_and_errors_per_day(method):
    r=analyze_research(ResearchRequest(samples=samples(),method=method))
    a=r['sources']['measured'];b=r['sources']['predicted']
    assert len(a['fit'])==48
    assert len(a['daily'])==2
    assert all(np.isfinite(a['fit']))
    for row in r['errors']:
        assert row['stec']['mae']==pytest.approx(1)
        assert row['stec']['rmse']==pytest.approx(1)
        assert row['stec']['bias']==pytest.approx(1)
        assert row['stec']['max_abs']==pytest.approx(1)
        assert row['delay_ns']['mae']==pytest.approx(40.3e16/(C*(1575.42e6)**2)*1e9)
        assert row['doppler_hz']['count']==(46 if row['day']=='all' else 23)
    assert r['errors'][-1]['stec']['count']==48
    if method=='harmonic':
        assert max(abs(np.array(a['residual'])))<1e-12
    assert a['series']['stec_tecu'][0]+1==b['series']['stec_tecu'][0]


def test_short_harmonic_fit_is_missing_rather_than_invented():
    r=analyze_research(ResearchRequest(samples=samples()[:3]))
    assert r['sources']['measured']['fit']==[None]*3
    assert r['sources']['measured']['daily'][0]['fit_note']


def test_acf_and_psd_do_not_stitch_gaps():
    times=np.r_[np.arange(12)*3600, np.arange(20,30)*3600]
    result=temporal_diagnostics(times,np.sin(times/3600))
    assert result['count']==12
    assert result['lags_s'][1]==3600
    assert max(result['frequency_hz'])<=1/7200


def test_frequency_derivative_matches_finite_difference():
    fc=1.57542e9;step=100
    f=lambda hz:40.3e16*25/(C*hz**2)
    expected=(f(fc+step)-f(fc-step))/(2*step)
    assert physical(np.array([25.]),fc)['delay_frequency_derivative_s_per_hz'][0]==pytest.approx(expected,rel=1e-8)


def test_pure_iono_response_preserves_energy_and_zero_tec_identity():
    zero=ionosphere_response(ResponseRequest(stec_tecu=0))
    assert np.allclose(zero['tx_real'],zero['rx_real'],atol=1e-12)
    r=ionosphere_response(ResponseRequest(stec_tecu=100,carrier_mhz=100,bandwidth_mhz=20))
    assert np.sum(r['tx_intensity'])==pytest.approx(np.sum(r['rx_intensity']),rel=1e-12)
    assert r['rx_rms_width_ns']>r['tx_rms_width_ns']
    with pytest.raises(ValueError):ionosphere_response(ResponseRequest(stec_tecu=1,carrier_mhz=100,bandwidth_mhz=200))


def test_lcr_counts_downward_crossings_and_afd_excludes_censored_fades():
    # At rho near 1, two completed fades of lengths 2 and 1, one leading censored fade.
    t=np.arange(10,dtype=float);amplitude=np.array([.2,2,2,.2,.2,2,2,.2,2,2])
    result=fading_statistics(t,amplitude,1)
    i=np.argmin(abs(np.array(result['thresholds_rms'])-1))
    assert result['lcr_hz'][i]==pytest.approx(2/9)
    assert result['afd_s'][i]==pytest.approx(1.5)
    assert result['complete_fades'][i]==2
    assert result['censored_fades'][i]==1
    constant=fading_statistics(np.arange(20)*.025,np.ones(20),0)
    assert all(v==0 for v in constant['lcr_theory_hz'])
    assert all(v is None for v in constant['afd_s'])


def test_joint_decomposition_reconstructs_each_observation():
    data=samples()
    for sample in data:
        hour=int(sample['time'][11:13]);sample['measured']+=.2*hour
    r=analyze_research(ResearchRequest(samples=data))
    source=r['sources']['measured'];decomposition=source['decomposition']
    reconstructed=np.asarray(decomposition['trend'])+np.asarray(decomposition['seasonal'])+np.asarray(decomposition['residual'])
    assert np.allclose(reconstructed,source['series']['stec_tecu'],atol=1e-12)
    assert r['errors'][-1]['stec']['interval95']==pytest.approx(np.quantile([s['predicted']-s['measured'] for s in data],[.025,.975]))


def test_profile_aggregates_actual_utc_times_across_days():
    r=analyze_research(ResearchRequest(samples=[
        {'time':'2024-01-01T00:00:00Z','measured':10},
        {'time':'2024-01-02T03:00:00+03:00','measured':20}]))
    p=r['sources']['measured']['hourly_profile']
    assert len(p)==1 and p[0]['count']==2
    assert p[0]['hour']==0
    assert p[0]['mean']==15
    assert p[0]['q25']==12.5 and p[0]['q75']==17.5
    assert r['sources']['measured']['decomposition']['trend']==[None,None]


def test_configurable_wave_count_duration_and_diagnostic_rate_are_used():
    from ml_lab.radio_simulator import RadioConfig, simulate, channel
    cfg=RadioConfig(model='combined',symbols=2048,sweep=False,tec_stec_tecu=20,
        tec_rate_tecu_s=.01,clarke_waves=32,diagnostic_duration_s=5,diagnostic_rate_hz=80)
    r=simulate(cfg);d=r['diagnostics'];t=np.asarray(d['time'])
    assert len(t)==400 and t[1]-t[0]==pytest.approx(1/80)
    expected=channel(cfg,t,cfg.seed+101)
    assert np.allclose(np.asarray(d['real'])+1j*np.asarray(d['imag']),expected)
    assert r['config']['clarke_waves']==32
    assert 1. in r['channel_statistics']['thresholds_rms']
    changed=simulate(cfg.model_copy(update={'clarke_waves':64}))
    assert not np.allclose(d['amplitude'],changed['diagnostics']['amplitude'])
