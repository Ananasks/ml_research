"""Physical invariants and independent statistical/reference checks."""
import json
import numpy as np
import pytest
from scipy.special import erfc, j0
from pydantic import ValidationError
from ml_lab.radio_simulator import RadioConfig, clarke, experiment, simulate


def test_clarke_ensemble_matches_power_correlation_and_rayleigh_moment():
    times = np.array([0., .07, .2])
    samples = np.array([clarke(times, 2, seed) for seed in range(2500)])
    assert np.mean(np.abs(samples[:, 0])**2) == pytest.approx(1, abs=.05)
    assert np.mean(np.abs(samples[:, 0])**4) == pytest.approx(2, abs=.15)
    covariance = np.mean(samples[:, 0, None]*samples.conj(), axis=0)
    assert np.max(np.abs(covariance.real-j0(2*np.pi*2*times))) < .05
    assert np.max(np.abs(covariance.imag)) < .05
    assert clarke(np.arange(10), 0, 42).std() < 1e-14


@pytest.mark.parametrize('mode',['narrow','wide'])
@pytest.mark.parametrize('modulation',['BPSK','QPSK','16-QAM'])
def test_noiseless_ideal_equalization_recovers_all_bits(mode, modulation):
    cfg = RadioConfig(mode=mode, modulation=modulation, rate_hz=1e6, noise_enabled=False, sweep=False)
    result = experiment(cfg)
    assert result['metrics']['errors'] == 0
    assert result['metrics']['evm_percent'] < 1e-9
    assert result['metrics']['interval95'][1] > 0


def test_awgn_ber_matches_independent_bpsk_expression():
    cfg = RadioConfig(model='clarke', fading_enabled=False, snr_db=0, modulation='BPSK', symbols=32768, sweep=False)
    result = experiment(cfg)
    assert result['metrics']['ber'] == pytest.approx(.5*erfc(1), abs=.006)
    assert result['rx_power'] == pytest.approx(1)
    assert result['noise_variance'] == pytest.approx(1)


def test_short_cp_creates_interference_even_with_exact_equalizer():
    cfg = RadioConfig(mode='wide', model='clarke', fading_enabled=False, rate_hz=1e6,
                      echo_delay_us=100, echo_db=0, noise_enabled=False, sweep=False)
    result = experiment(cfg)
    assert not result['cp_sufficient']
    assert result['metrics']['evm_percent'] > 10
    assert result['metrics']['errors'] > 0


def test_export_reproducibility_finite_and_zero_doppler_acf():
    cfg = RadioConfig(mode='wide', doppler_hz=0, seed=4)
    first, second = simulate(cfg), simulate(cfg)
    assert json.dumps(first, allow_nan=False) == json.dumps(second, allow_nan=False)
    assert all(value is None for value in first['diagnostics']['acf'])


def test_invalid_requests_are_rejected():
    for changes in ({'symbols':10}, {'seed':-1}, {'mode':'unknown'}, {'doppler_hz':float('nan')},
                    {'mode':'wide','rate_hz':20e6,'echo_delay_us':500}):
        with pytest.raises(ValidationError):
            RadioConfig(**changes)


def test_error_bursts_and_block_measurements_have_known_answer():
    from ml_lab.radio_simulator import error_statistics, constellation
    points, labels = constellation('BPSK')
    tx = np.zeros(256, dtype=int)
    rx = points[tx].copy()
    rx[[1, 2, 3, 20, 128, 129]] = points[1]
    result = error_statistics(tx, rx, points, labels)
    assert result['burst_lengths'] == [1, 2, 3]
    assert result['burst_counts'] == [1, 1, 1]
    assert result['max_burst_bits'] == 3
    assert result['block_count'] == 2
    assert result['block_error_rate'] == 1
    assert result['error_acf'][0] == pytest.approx(1)
    zero = error_statistics(tx, points[tx], points, labels)
    assert zero['burst_count'] == 0
    assert zero['block_error_rate'] == 0
    assert all(v is None for v in zero['error_acf'])


def test_independent_models_are_available_without_blank():
    from ml_lab.radio_channels import MODEL_REGISTRY
    assert set(MODEL_REGISTRY)=={'tec','clarke','combined'}
    assert RadioConfig().model=='clarke'
    with pytest.raises(ValidationError):
        RadioConfig(model='blank')
