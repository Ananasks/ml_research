import math
import pytest
from ml_lab.channel_setup import ChannelSetupRequest, prepare_channel_setup
from ml_lab.channel_lab import ChannelLabConfig, run_channel_lab


def prepare(config, **options):
    return prepare_channel_setup(ChannelSetupRequest(config=config,**options))


@pytest.mark.parametrize('model',['clarke','screen','physical'])
def test_auto_defaults_are_valid_and_have_parameter_provenance(model):
    p=prepare({'model':model})
    cfg=ChannelLabConfig.model_validate(p['config'])
    assert cfg.duration_s==30
    assert p['parameters']
    assert {r['origin'] for r in p['parameters']} <= {'input','formula','scenario','numerical'}
    run_channel_lab(cfg)


def test_clarke_sampling_changes_with_key_doppler_and_handles_budget():
    a=prepare({'model':'clarke','doppler_hz':1})
    b=prepare({'model':'clarke','doppler_hz':5})
    c=prepare({'model':'clarke','doppler_hz':30})
    assert a['config']['sample_rate_hz']==40
    assert b['config']['sample_rate_hz']==100
    assert c['config']['sample_rate_hz']==160
    assert c['config']['sample_rate_hz']>2*c['config']['doppler_hz']
    assert any('160' in note for note in c['notes'])


def test_auto_resolves_invalid_manual_sampling_but_manual_rejects_it():
    config={'model':'clarke','doppler_hz':10,'sample_rate_hz':10}
    assert prepare(config)['config']['sample_rate_hz']>20
    with pytest.raises(ValueError):
        prepare(config,setup_mode='manual')


def test_screen_grid_satisfies_resolution_and_finite_track_conditions():
    p=prepare({'model':'screen','phase_rms_rad':.8})
    c=p['config']
    assert c['phase_rms_rad']==.8
    assert c['screen_size'] in (128,256)
    assert c['spacing_m']<=c['inner_scale_m']/2
    assert c['screen_size']*c['spacing_m']>=4*c['outer_scale_m']
    assert c['velocity_m_s']*c['duration_s']<=.8*c['screen_size']*c['spacing_m']


def test_mean_tec_does_not_imply_phase_fluctuations():
    a=prepare({'model':'physical','stec_tecu':2},scenario='quiet')
    b=prepare({'model':'physical','stec_tecu':20},scenario='quiet')
    assert a['config']['phase_rms_rad']==b['config']['phase_rms_rad']
    assert a['config']['outer_scale_m']==b['config']['outer_scale_m']
    by_key=lambda p:{r['key']:r for r in p['parameters']}
    assert by_key(b)['delay_ns']['value']==pytest.approx(10*by_key(a)['delay_ns']['value'])
    assert by_key(a)['phase_rms_rad']['origin']=='scenario'


def test_vtec_conversion_is_applied_once_and_transforms_local_derivative():
    a=prepare({'model':'physical','scattering':'none','stec_tecu':2,'tec_rate_tecu_s':.001},tec_kind='vtec',elevation_deg=30)
    m=(1-(6371*math.cos(math.pi/6)/(6371+350))**2)**-.5
    assert a['config']['stec_tecu']==pytest.approx(2*m)
    assert a['config']['tec_rate_tecu_s']==pytest.approx(.001*m)
    b=prepare(a['config'],tec_kind='stec')
    assert b['config']['stec_tecu']==pytest.approx(2*m)


def test_manual_preserves_custom_screen_parameters():
    p=prepare({'model':'screen','phase_rms_rad':.7,'outer_scale_m':400,'inner_scale_m':30,'spacing_m':10,'screen_size':256},setup_mode='manual')
    assert p['config']['outer_scale_m']==400
    assert p['config']['phase_rms_rad']==.7
    assert p['config']['screen_size']==256


def test_extreme_track_is_rejected_with_actionable_message():
    with pytest.raises(ValueError,match='Уменьшите длительность'):
        prepare({'model':'screen','carrier_mhz':12000,'distance_km':100,'duration_s':120},scenario='disturbed')


def test_zero_distance_has_explicit_scenario_fallback():
    p=prepare({'model':'screen','distance_km':0})
    assert any('100 м' in note for note in p['notes'])
    assert p['config']['distance_km']==0
