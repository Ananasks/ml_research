import numpy as np
import pandas as pd
import pytest
from ml_lab.tec_sources import prepare_tec, summarize_pair, serialize_series
from ml_lab.radio_channels.tec import physical, mapping, transfer, C
from ml_lab.radio_simulator import RadioConfig, experiment, simulate


def data(times=None, values=None):
    return pd.DataFrame({'utc':times or ['2026-01-01T00:00Z','2026-01-01T00:01Z','2026-01-01T00:02Z'], 'tec':values or [10,11,12]})


def test_physical_units_frequency_scaling_and_phase_sign():
    f=1575.42e6
    r=physical([10],f,1e6)
    assert r['path_m'][0]==pytest.approx(1.6237245,rel=1e-5)
    assert physical([10],f/2)['delay_ns'][0]==pytest.approx(r['delay_ns'][0]*4)
    delta=100
    derivative=(np.angle(transfer(10,f+delta)/transfer(10,f-delta)))/(2*delta)
    assert -derivative/(2*np.pi)==pytest.approx(r['delay_ns'][0]*1e-9,rel=1e-7)
    assert r['dispersion_ns'][0]>0
    assert mapping(90)==pytest.approx(1)


def test_unit_conversion_utc_alignment_and_errors():
    real, q=prepare_tec(data(),'utc','tec',kind='STEC')
    other=data(['2026-01-01T03:00+03:00','2026-01-01T03:01+03:00','2026-01-01T03:02+03:00'],[11e16,12e16,13e16])
    pred,_=prepare_tec(other,'utc','tec',unit='electrons/m2',kind='STEC')
    pair,metrics=summarize_pair(real,pred,1e9,1e6)
    assert len(pair)==3
    assert metrics['stec']['rmse']==pytest.approx(1)
    assert metrics['delay']['mae']==pytest.approx(40.3e16/(C*1e18)*1e9)
    assert real.rate.iloc[1]==pytest.approx(1/60)
    assert serialize_series(real,1e9,1e6)['doppler_hz'][0] is None


def test_duplicate_invalid_negative_missing_and_no_overlap():
    with pytest.raises(ValueError,match='дубликаты'):
        prepare_tec(data(['2026-01-01','2026-01-01','2026-01-02']),'utc','tec')
    with pytest.raises(ValueError,match='отрицательным'):
        prepare_tec(data(values=[-1,2,3]),'utc','tec')
    with pytest.raises(ValueError,match='неоднозначно'):
        prepare_tec(data([1,2,3]),'utc','tec')
    frame=data();frame.loc[0,'tec']=np.nan
    clean,q=prepare_tec(frame,'utc','tec')
    assert q['discarded_rows']==1
    clean2,_=prepare_tec(data(['2025-01-01','2025-01-02','2025-01-03']),'utc','tec')
    with pytest.raises(ValueError,match='общих'):
        summarize_pair(clean,clean2,1e9,1e6)


def test_derivative_not_estimated_across_large_gap():
    frame=pd.DataFrame({'utc':pd.to_datetime([0,60,120,3600,3660],unit='s',utc=True),'tec':[10,11,12,30,31]})
    clean,q=prepare_tec(frame,'utc','tec')
    assert q['gaps']==1
    assert np.isnan(clean.rate.iloc[3])


@pytest.mark.parametrize('mode',['narrow','wide'])
def test_tec_noiseless_ideal_vs_forecast_compensation(mode):
    cfg=RadioConfig(mode=mode,model='tec',tec_stec_tecu=35,noise_enabled=False,echo_enabled=False,rate_hz=1e6,sweep=False)
    assert experiment(cfg)['metrics']['errors']==0
    perfect=cfg.model_copy(update={'channel_estimation':'forecast','compensation_stec_tecu':35})
    assert experiment(perfect)['metrics']['errors']==0
    wrong=perfect.model_copy(update={'compensation_stec_tecu':35.25})
    assert experiment(wrong)['metrics']['evm_percent']>30


@pytest.mark.parametrize('mode',['narrow','wide'])
def test_pilot_receiver_excludes_known_symbols_and_recovers_constant_channel(mode):
    cfg=RadioConfig(mode=mode,model='tec',tec_stec_tecu=35,noise_enabled=False,echo_enabled=False,channel_estimation='pilot',sweep=False)
    r=experiment(cfg)
    assert r['pilot_symbols']==cfg.symbols//16
    assert r['metrics']['bits']==(cfg.symbols-cfg.symbols//16)*2
    assert r['metrics']['errors']==0


def test_independent_realization_seeds_and_interval_reproducible():
    cfg=RadioConfig(realizations=3,sweep=False)
    first=simulate(cfg)['ensemble'];second=simulate(cfg)['ensemble']
    assert first==second
    assert first['seeds']==[42,43,44]
    assert len(first['interval95'])==2


def test_date_and_time_adapter_reuses_dataset_parser():
    from api.tec import AnalysisRequest, Source, resolve
    request=AnalysisRequest(measured=Source(dataset='tec_cb.csv',time_column='__auto__',value_column='kabc_absTEC'))
    clean,metadata=resolve(request.measured,request)
    assert clean.time.iloc[0].isoformat()=='2022-01-01T00:00:00+00:00'
    assert clean.time.iloc[1]-clean.time.iloc[0]==pd.Timedelta(minutes=30)
    assert metadata['quality']['discarded_rows']>0


def test_combined_measurement_forecast_experiment_uses_same_seed():
    cfg=RadioConfig(model='combined',tec_stec_tecu=35,compensation_stec_tecu=35,
                    source_context={'input_mode':'compare'},sweep=False)
    result=simulate(cfg)
    assert result['primary']==result['predicted_channel']


def test_configurable_ofdm_and_noiseless_pilot_receiver():
    cfg=RadioConfig(mode='wide',nfft=2048,cp_divisor=8,model='tec',tec_stec_tecu=35,
                    echo_enabled=False,noise_enabled=False,channel_estimation='pilot',rate_hz=1e6,sweep=False)
    result=experiment(cfg)
    assert result['taps']['cp']==256
    assert result['metrics']['errors']==0


def test_full_saved_ml_predictions_are_used_without_preview_downsampling(monkeypatch):
    from api import main
    from api.tec import AnalysisRequest, Source, resolve
    n=9000
    frame=pd.DataFrame({'Время':pd.date_range('2026-01-01',periods=n,freq='min'),
                        'Факт':np.full(n,20.),'Прогноз':np.full(n,21.)})
    monkeypatch.setattr(main,'RUNS',{'fixture':{'task':'Регрессия','name':'Test regression','target':'VTEC','predictions':frame}})
    source=Source(kind='run',run_id='fixture',run_field='Прогноз',tec_type='STEC')
    clean,metadata=resolve(source,AnalysisRequest(predicted=source))
    assert len(clean)==n
    assert metadata['full_saved_predictions']
    assert (clean.stec==21).all()


def test_saved_run_picker_marks_missing_time_and_excludes_classification(monkeypatch,tmp_path):
    from api import main
    from api.tec import available_tec_runs
    class Store(dict):
        directory=tmp_path
    store=Store({
        'dated':{'task':'Регрессия','name':'Ridge','target':'VTEC','predictions':pd.DataFrame({'Время':pd.date_range('2026-01-01',periods=2),'Прогноз':[20,21]})},
        'undated':{'task':'Регрессия','name':'Ridge','target':'VTEC','predictions':pd.DataFrame({'Прогноз':[20,21]})},
        'classifier':{'task':'Классификация','predictions':pd.DataFrame({'Прогноз':[0,1]})},
    })
    for key in store:
        (tmp_path/f'{key}.joblib').touch()
    monkeypatch.setattr(main,'RUNS',store)
    options={row['id']:row for row in available_tec_runs()}
    assert set(options)=={'dated','undated'}
    assert options['dated']['timeAvailable']
    assert not options['undated']['timeAvailable']


def test_ofdm_waveform_comparison_uses_matching_useful_samples_without_cp():
    cfg=RadioConfig(mode='wide',model='tec',tec_stec_tecu=0,noise_enabled=False,
                    echo_enabled=False,rate_hz=1e6,sweep=False)
    wave=experiment(cfg)['waveform']
    assert len(wave['tx_real'])==len(wave['real'])==len(wave['time_s'])
    assert np.allclose(wave['tx_real'],wave['real'],atol=1e-12)


def test_dispersive_bandwidth_scaling_and_zero_serialization():
    f=1e9
    r=physical([10,40],f)
    expected=np.sqrt(C*f**3/(80.5*np.pi*10e16))/1e6
    assert r['coherence_bandwidth_mhz'][0]==pytest.approx(expected)
    assert r['coherence_bandwidth_mhz'][1]==pytest.approx(expected/2)
    assert physical([10],2*f)['coherence_bandwidth_mhz'][0]==pytest.approx(expected*2**1.5)
    assert r['delay_dispersion_s2'][0]==pytest.approx(-80.5*10e16/(2*np.pi*C*f**3))
    frame,_=prepare_tec(data(values=[0,10,40]),'utc','tec',kind='STEC')
    serialized=serialize_series(frame,f,1e6)
    assert serialized['coherence_bandwidth_mhz'][0] is None
    assert serialized['delay_ns'][0]==0


def test_residual_bandwidth_uses_absolute_error_at_common_times(monkeypatch):
    from api import tec
    measured,_=prepare_tec(data(),'utc','tec',kind='STEC')
    predicted,_=prepare_tec(data(values=[10,9,14]),'utc','tec',kind='STEC')
    monkeypatch.setattr(tec,'resolve',lambda source,request: (measured if source.dataset=='real' else predicted,{}))
    result=tec.analyze(tec.AnalysisRequest(measured=tec.Source(dataset='real'),predicted=tec.Source(dataset='pred')))
    assert result['residual']['stec_tecu']==[0,2,2]
    assert result['residual']['coherence_bandwidth_mhz'][0] is None
    assert result['residual']['coherence_bandwidth_mhz'][1]==pytest.approx(result['residual']['coherence_bandwidth_mhz'][2])
    import json
    json.dumps(result,allow_nan=False)
