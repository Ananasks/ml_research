"""Signal-path invariants, using an independently specified channel operator."""
import numpy as np
import pytest
from ml_lab.channel_sounding import sound_channel
from ml_lab.channel_lab import ChannelLabConfig, run_channel_lab


def test_identity_recovers_payload_and_training_without_noise():
    r = sound_channel(np.ones((129, 4), complex), np.arange(-64,65)*1000,
                      np.arange(4), seed=7, snr_db=20, noise_enabled=False)
    assert r['receiver']['errors'] == 0
    assert r['receiver']['evm_percent'] < 1e-10
    assert r['verification']['channel_nmse'] < 1e-28
    assert r['verification']['parseval_error'] < 1e-14
    np.testing.assert_allclose(r['waveform']['tx']['i'], r['waveform']['rx']['i'], atol=1e-14)


def test_delay_has_expected_cir_and_training_not_payload_noise():
    n, delay = 129, 5
    frequency = np.arange(-64,65)*1000
    h = np.exp(-2j*np.pi*frequency*delay/(n*1000))[:,None]*np.ones((1,128))
    clean = sound_channel(h, frequency, np.arange(128), seed=3, snr_db=20, noise_enabled=False)
    assert np.argmax(clean['truth']['cir_power']) == n//2+delay
    assert clean['receiver']['errors'] == 0
    noisy = sound_channel(h, frequency, np.arange(128), seed=3, snr_db=0, noise_enabled=True)
    assert noisy['receiver']['ber'] > noisy['ideal_reference']['ber']
    assert noisy['verification']['channel_nmse'] == pytest.approx(1, abs=.03)


@pytest.mark.parametrize('model', ['clarke','screen','physical'])
@pytest.mark.parametrize('mode', ['narrow','wide'])
def test_every_model_transmits_same_source_with_separate_truth_and_estimate(model, mode):
    cfg = ChannelLabConfig(model=model, mode=mode, duration_s=2, noise_enabled=False)
    r = run_channel_lab(cfg)
    assert r['transmission']['receiver']['errors'] == 0
    assert r['transmission']['verification']['channel_nmse'] < 1e-28
    baseline = run_channel_lab(cfg.model_copy(update={'model':'clarke'}))
    assert r['transmission']['waveform']['tx'] == baseline['transmission']['waveform']['tx']


@pytest.mark.parametrize('signal_type', ['ofdm', 'qpsk', 'gnss', 'lfm'])
@pytest.mark.parametrize('propagation', ['circular', 'linear'])
def test_all_sources_through_identity(signal_type, propagation):
    r = sound_channel(np.ones((129, 8), complex), np.arange(-64,65)*1000,
                      np.arange(8), seed=7, snr_db=20, noise_enabled=False,
                      signal_type=signal_type, propagation=propagation, cp_samples=0)
    assert r['receiver']['evm_percent'] < 1e-10
    assert r['receiver']['ber'] == (None if signal_type=='lfm' else 0)
    assert r['numerical_latency_samples'] == 0


def test_linear_memory_and_prefix_shortfall_against_two_tap_channel():
    n = 129
    f = np.arange(-64,65)*1000
    h = (1+.7*np.exp(-2j*np.pi*f*12/(n*1000)))[:,None]*np.ones((1,64))
    def run(cp):
        return sound_channel(h, f, np.arange(64), seed=5, snr_db=20, noise_enabled=False,
                             propagation='linear', cp_samples=cp)
    sufficient, short = run(12), run(0)
    assert sufficient['receiver']['evm_percent'] < 1e-10
    assert sufficient['verification']['channel_nmse'] < 1e-28
    assert short['ideal_reference']['evm_percent'] > 10
    assert short['verification']['channel_nmse'] > .01
    assert short['cp_tail_energy_fraction'] == pytest.approx(.49/1.49)
    assert sufficient['cp_tail_energy_fraction'] < 1e-28


def test_negative_relative_delay_is_shifted_and_reported_without_axis_corruption():
    f = np.arange(-64,65)*1000
    h = np.exp(2j*np.pi*f*7/(129*1000))[:,None]*np.ones((1,4))
    r = sound_channel(h, f, np.arange(4), seed=5, snr_db=20, noise_enabled=False,
                      propagation='linear', cp_samples=0)
    assert r['numerical_latency_samples'] == 7
    assert np.argmax(r['truth']['cir_power']) == 64-7
    assert r['receiver']['evm_percent'] < 1e-10


def test_coherence_is_finite_band_and_distinguishes_noise_from_truth():
    f = np.arange(-64,65)*1000
    r = sound_channel(np.ones((129,64), complex), f, np.arange(64), seed=9,
                      snr_db=-5, noise_enabled=True)
    assert r['truth']['coherence']['bc_hz'] is None
    assert r['estimated']['coherence']['bc_hz'] is not None
    assert r['estimated']['coherence']['correlation'][0] == pytest.approx(1)
