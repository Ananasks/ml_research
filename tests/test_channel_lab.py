"""Physical limiting cases and spectral identities of the channel lab."""
import numpy as np
import pytest
from pydantic import ValidationError
from ml_lab.channel_lab import ChannelLabConfig, phase_screen, propagate, run_channel_lab, C


def test_zero_screen_is_unperturbed_plane_wave():
    cfg = ChannelLabConfig(model='screen', phase_rms_rad=0, duration_s=2)
    result = run_channel_lab(cfg)
    np.testing.assert_allclose(result['amplitude'], 1, atol=1e-14)
    assert result['metrics']['s4'] < 1e-12
    assert all(v is None for v in result['acf'])


def test_screen_at_zero_distance_has_no_intensity_scintillation_on_grid():
    cfg = ChannelLabConfig(model='screen', distance_km=0)
    phi = phase_screen(cfg)
    field = propagate(phi, cfg.carrier_mhz*1e6, cfg)
    np.testing.assert_allclose(abs(field), 1, atol=1e-14)
    r = run_channel_lab(cfg)
    np.testing.assert_allclose(r['amplitude'], 1, atol=1e-14)


def test_fresnel_propagation_conserves_spatial_energy():
    cfg = ChannelLabConfig(model='screen', phase_rms_rad=1.2)
    phi = phase_screen(cfg)
    np.testing.assert_allclose(phi.std(), 1.2, atol=1e-12)
    field = propagate(phi, cfg.carrier_mhz*1e6, cfg)
    assert abs(np.mean(abs(field)**2)-1) < 1e-12
    assert np.std(abs(field)**2) > .01


def test_fixed_receiver_produces_constant_time_series():
    r = run_channel_lab(ChannelLabConfig(model='screen', velocity_m_s=0, duration_s=2))
    np.testing.assert_allclose(r['amplitude'], r['amplitude'][0])
    assert r['metrics']['s4'] < 1e-12


def test_pure_tec_path_phase_and_frequency_follow_cold_plasma():
    cfg = ChannelLabConfig(model='physical', scattering='none', stec_tecu=2.336, tec_rate_tecu_s=.01, duration_s=2)
    r = run_channel_lab(cfg)
    fc = cfg.carrier_mhz*1e6
    path = 40.3*2.336e16/fc**2
    assert r['physical']['path_m'][0] == pytest.approx(path)
    assert r['physical']['delay_ns'][0] == pytest.approx(path/C*1e9)
    assert path == pytest.approx(.379, abs=.001)
    phase = np.array(r['physical']['phase_rad'])
    np.testing.assert_allclose(np.diff(phase)*cfg.sample_rate_hz/(2*np.pi), r['physical']['doppler_hz'], rtol=1e-10)
    np.testing.assert_allclose(r['amplitude'], 1, atol=1e-14)


def test_delay_scales_inverse_square_and_phase_inverse_frequency():
    a = run_channel_lab(ChannelLabConfig(model='physical', scattering='none', duration_s=2, carrier_mhz=1000))
    b = run_channel_lab(ChannelLabConfig(model='physical', scattering='none', duration_s=2, carrier_mhz=2000))
    assert a['physical']['delay_ns'][0]/b['physical']['delay_ns'][0] == pytest.approx(4)
    assert a['physical']['phase_rad'][0]/b['physical']['phase_rad'][0] == pytest.approx(2)


def test_clarke_single_path_is_frequency_flat():
    r = run_channel_lab(ChannelLabConfig(model='clarke', mode='wide', paths=1, duration_s=2))
    np.testing.assert_allclose(r['gain_db'], r['gain_db'][0], atol=1e-12)
    np.testing.assert_allclose(r['correlation']['value'], 1, atol=1e-12)
    assert r['metrics']['bc_mhz'] is None


def test_clarke_zero_delay_multipath_is_flat_and_psd_nonnegative():
    r = run_channel_lab(ChannelLabConfig(model='clarke', mode='wide', paths=4, delay_span_us=0, duration_s=2))
    np.testing.assert_allclose(r['gain_db'], r['gain_db'][0], atol=1e-12)
    np.testing.assert_allclose(r['correlation']['value'], 1, atol=1e-12)
    assert sum(r['taps']['power']) == pytest.approx(1)
    assert np.isfinite(r['psd']['density']).all()
    assert np.min(r['psd']['density']) >= 0


def test_reproducible_screen_and_distinct_random_seeds():
    a = phase_screen(ChannelLabConfig(model='screen', seed=12))
    b = phase_screen(ChannelLabConfig(model='screen', seed=12))
    c = phase_screen(ChannelLabConfig(model='screen', seed=13))
    np.testing.assert_array_equal(a,b)
    assert not np.allclose(a,c)


@pytest.mark.parametrize('options', [dict(sample_rate_hz=10,doppler_hz=5),dict(model='screen',velocity_m_s=500),dict(inner_scale_m=500,outer_scale_m=100),dict(stec_tecu=0,tec_rate_tecu_s=-1)])
def test_invalid_or_unresolved_experiments_rejected(options):
    with pytest.raises(ValidationError):
        ChannelLabConfig(**options)


def test_wide_screen_frequency_response_is_finite_and_nonflat():
    r = run_channel_lab(ChannelLabConfig(model='screen', mode='wide', bandwidth_mhz=20, duration_s=2))
    assert np.isfinite(r['gain_db']).all()
    assert np.std(r['gain_db']) > 1e-5
    assert r['screen']['mean_intensity'] == pytest.approx(1)
    assert np.min(r['cir_power']) >= 0


def test_cir_parseval_and_frequency_origin():
    r = run_channel_lab(ChannelLabConfig(model='clarke', mode='wide', duration_s=2))
    assert r['frequency_mhz'][64] == pytest.approx(0)
    mean_gain_power = np.mean(10**(np.array(r['gain_db'])/10))
    assert sum(r['cir_power']) == pytest.approx(mean_gain_power)


def test_api_accepts_new_model_and_rejects_aliasing():
    import asyncio
    import json
    from fastapi import FastAPI
    from api.radio import router
    app = FastAPI()
    app.include_router(router)

    async def post(payload):
        body = json.dumps(payload).encode()
        messages = []
        async def receive():
            return {'type':'http.request','body':body,'more_body':False}
        async def send(message):
            messages.append(message)
        await app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1',
                   'method':'POST','scheme':'http','path':'/api/radio/channel-lab',
                   'raw_path':b'/api/radio/channel-lab','query_string':b'',
                   'root_path':'','headers':[(b'content-type',b'application/json')],
                   'server':('test',80),'client':('test',1234)},receive,send)
        return messages[0]['status'], b''.join(m.get('body',b'') for m in messages[1:])

    status, body = asyncio.run(post({'model':'physical','scattering':'none','duration_s':2}))
    assert status == 200
    assert len(json.loads(body)['time_s']) == 80
    status, _ = asyncio.run(post({'sample_rate_hz':10,'doppler_hz':10}))
    assert status == 422
