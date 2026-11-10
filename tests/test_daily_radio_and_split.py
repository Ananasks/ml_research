import numpy as np
import pandas as pd
import pytest
from ml_lab.core import split_data
from ml_lab.data import prepare_frame
from ml_lab.radio_daily import DailyRequest, compare_daily
from ml_lab.radio_simulator import RadioConfig, experiment


def test_day_range_uses_real_dates_with_irregular_counts_and_inclusive_end():
    times = pd.date_range('2024-01-01', periods=12*24, freq='h').delete([99, 100, 151])
    raw = pd.DataFrame({'datetime': times, 'x': np.arange(len(times)), 'y': np.arange(len(times))})
    data = prepare_frame(raw, target='y')
    X, _, (tr, va, te) = split_data(data, ['x'], 'y', 'Регрессия', {
        'size_mode': 'days', 'split': 'time', 'day_start': '2024-01-03',
        'day_end': '2024-01-10', 'validation_days': 2, 'test_days': 2})
    assert set(X.iloc[tr].index.day) == {3, 4, 5, 6}
    assert set(X.iloc[va].index.day) == {7, 8}
    assert set(X.iloc[te].index.day) == {9, 10}
    assert X.iloc[te].index[-1].hour == 23


def test_day_split_purges_forecast_overlap():
    raw = pd.DataFrame({'datetime': pd.date_range('2024-01-01', periods=240, freq='h'),
                        'x': np.arange(240), 'y': np.arange(240)})
    data = prepare_frame(raw, target='y', horizon=4)
    X, _, (tr, va, te) = split_data(data, ['x'], 'y', 'Регрессия', {
        'split': 'time', 'size_mode': 'days', 'validation_days': 2, 'test_days': 2})
    assert X.iloc[tr].index[-1] + pd.Timedelta(hours=4) < X.iloc[va].index[0]
    assert X.iloc[va].index[-1] + pd.Timedelta(hours=4) < X.iloc[te].index[0]


@pytest.mark.parametrize('options', [{'split': 'random'}, {'test_days': 30}, {'test_days': 1.5},
                                   {'day_start': '2024-02-01', 'day_end': '2024-01-01'}])
def test_day_split_rejects_invalid_ranges_and_counts(options):
    data = prepare_frame(pd.DataFrame({'datetime': pd.date_range('2024-01-01', periods=240, freq='h'),
                                     'x': np.arange(240), 'y': np.arange(240)}), target='y')
    with pytest.raises(ValueError):
        split_data(data, ['x'], 'y', 'Регрессия', {'split': 'time', 'size_mode': 'days', **options})


@pytest.mark.parametrize('mode', ['narrow', 'wide'])
def test_three_daily_cases_match_packets_and_aggregate_bit_counts(mode):
    cfg = RadioConfig(mode=mode, symbols=2048, sweep=False, rate_hz=1e6)
    request = DailyRequest(config=cfg, samples=[
        {'time': '2024-01-01T23:00:00Z', 'measured': 20, 'predicted': 20},
        {'time': '2024-01-02T00:00:00Z', 'measured': 30, 'predicted': 31}])
    result = compare_daily(request)
    assert len(result['rows']) == 6
    assert result['rows'][0]['metrics'] == result['rows'][2]['metrics']
    for case in ['measured', 'clarke', 'predicted']:
        rows = [r for r in result['rows'] if r['case'] == case]
        total = next(r for r in result['summaries'] if r['day'] == 'all' and r['case'] == case)
        assert total['bits'] == sum(r['metrics']['bits'] for r in rows)
        assert total['ber'] == sum(r['metrics']['errors'] for r in rows)/total['bits']
    assert len({r['seed'] for r in result['rows'][:3]}) == 1
    assert result == compare_daily(request)


def test_daily_utc_missing_forecast_and_duplicate_handling():
    cfg = RadioConfig(symbols=2048, sweep=False)
    result = compare_daily(DailyRequest(config=cfg, samples=[
        {'time': '2024-01-02T01:00:00+03:00', 'measured': 10}]))
    assert {r['day'] for r in result['rows']} == {'2024-01-01'}
    assert {r['case'] for r in result['rows']} == {'measured', 'clarke'}
    with pytest.raises(ValueError, match='уникальны'):
        compare_daily(DailyRequest(config=cfg, samples=[{'time': '2024-01-01T00:00:00Z', 'measured': 1}]*2))


def test_compact_packet_preserves_metrics():
    cfg = RadioConfig(symbols=2048, sweep=False)
    full, compact = experiment(cfg), experiment(cfg, compact=True)
    assert all(compact[key] == full[key] for key in ['metrics', 'raw_metrics'])
    assert compact['metrics']['evm_percent'] == pytest.approx(100*np.sqrt(compact['error_energy']/compact['reference_energy']))


def test_qam_daily_evm_combines_actual_reference_energies():
    result = compare_daily(DailyRequest(config=RadioConfig(modulation='16-QAM', symbols=2048, sweep=False),
        samples=[{'time': '2024-01-01T00:00:00Z', 'measured': 20},
                 {'time': '2024-01-01T01:00:00Z', 'measured': 30}]))
    rows = [r for r in result['rows'] if r['case'] == 'clarke']
    total = next(r for r in result['summaries'] if r['day'] == 'all' and r['case'] == 'clarke')
    assert total['evm_percent'] == pytest.approx(100*np.sqrt(sum(r['error_energy'] for r in rows)/sum(r['reference_energy'] for r in rows)))


@pytest.mark.parametrize('mode',['narrow','wide'])
def test_second_scenario_multiplies_measured_tec_by_clarke(mode):
    from datetime import datetime
    sample={'time':'2024-01-01T12:00:00Z','measured':35,'predicted':34,'measured_rate':.02}
    config=RadioConfig(mode=mode,symbols=2048,rate_hz=1e6,sweep=False)
    result=compare_daily(DailyRequest(config=config,samples=[sample]))
    second=next(row for row in result['rows'] if row['case']=='clarke')
    seed=(config.seed+int(datetime.fromisoformat(sample['time'].replace('Z','+00:00')).timestamp()))%2147483648
    expected=experiment(config.model_copy(update={'model':'combined','tec_stec_tecu':35,'tec_rate_tecu_s':.02,
        'compensation_stec_tecu':34,'seed':seed}),compact=True)
    assert second['model']=='combined'
    assert second['stec_tecu']==35
    assert second['metrics']==expected['metrics']
    assert second['raw_metrics']==expected['raw_metrics']


def test_predicted_only_sample_does_not_create_a_measured_hybrid():
    result=compare_daily(DailyRequest(config=RadioConfig(symbols=2048,sweep=False),samples=[
        {'time':'2024-01-01T00:00:00Z','predicted':20}]))
    assert [row['case'] for row in result['rows']]==['predicted']


def test_hybrid_diagnostics_include_physical_frequency_shift():
    from ml_lab.radio_simulator import simulate
    from ml_lab.radio_channels.tec import C
    config=RadioConfig(model='combined',carrier_mhz=100,tec_stec_tecu=35,tec_rate_tecu_s=.1,
                       doppler_hz=0,symbols=2048,sweep=False)
    result=simulate(config)
    d=result['diagnostics']
    frequency=np.array(d['channel_psd']['f'])
    peak=frequency[np.argmax(d['channel_psd']['db'])]
    expected=40.3e16*.1/(C*100e6)
    assert peak==pytest.approx(expected,abs=.08)
    assert all(v is None for v in d['acf'])  # Constant statistical multiplier.
    assert any(v is not None and v<0 for v in d['channel_acf'])
