"""Deterministic parameter preparation; no random fields or channel simulation."""
from typing import Literal
import math
from pydantic import BaseModel, Field
from ml_lab.channel_lab import C, REFERENCE_HZ, ChannelLabConfig


class ChannelSetupRequest(BaseModel):
    config: dict
    setup_mode: Literal['automatic','manual'] = 'automatic'
    scenario: Literal['quiet','moderate','disturbed'] = 'moderate'
    tec_kind: Literal['stec','vtec'] = 'stec'
    elevation_deg: float = Field(45, ge=5, le=90)
    layer_height_km: float = Field(350, ge=100, le=1000)


# Engineering scenarios, not prescribed ITU profiles or consequences of mean TEC.
SCENARIOS = {
    'quiet': dict(phase=.2, doppler=.2, velocity=20., alpha=4., beta=.4, paths=2, delay=.5),
    'moderate': dict(phase=.5, doppler=1., velocity=40., alpha=4., beta=.4, paths=4, delay=2.),
    'disturbed': dict(phase=1., doppler=3., velocity=60., alpha=4., beta=.4, paths=6, delay=5.),
}


def prepare_channel_setup(request: ChannelSetupRequest):
    raw = dict(request.config)
    # Validate physical inputs and numeric bounds without enforcing relationships
    # that the automatic planner is about to resolve.
    values = {}
    for name, field in ChannelLabConfig.model_fields.items():
        if name in raw:
            from pydantic import TypeAdapter
            values[name] = TypeAdapter(field.rebuild_annotation()).validate_python(raw[name])
        else:
            values[name] = field.get_default(call_default_factory=True)
    if values['source_context'].get('kind') in ('measured', 'predicted') and (
        values['source_context'].get('derivative_available') is False or 'tec_rate_tecu_s' not in raw
    ):
        raise ValueError('Для измеренного или прогнозного ПЭС нужна известная производная; нулевая производная не подставляется автоматически.')
    scenario = SCENARIOS[request.scenario]
    rows = []
    notes = []
    def record(key, title, value, unit, origin, rule):
        rows.append(dict(key=key,title=title,value=value,unit=unit,origin=origin,rule=rule))
    physical = values['model']=='physical'
    scattering = values['scattering'] if physical else values['model']
    automatic = request.setup_mode=='automatic'
    if physical:
        mapping = 1.
        if request.tec_kind=='vtec':
            earth = 6371.
            mapping = (1-(earth*math.cos(math.radians(request.elevation_deg))/(earth+request.layer_height_km))**2)**-.5
            values['stec_tecu'] *= mapping
            values['tec_rate_tecu_s'] *= mapping
        record('mapping','Наклонный множитель M(E)',mapping,'','formula','[1−(RE cos E/(RE+hI))²]⁻¹ᐟ²; для входного STEC M=1')
        record('stec_tecu','STEC',values['stec_tecu'],'TECU','formula' if request.tec_kind=='vtec' else 'input','STEC=M(E)·VTEC' if request.tec_kind=='vtec' else 'Входное наклонное ПЭС, без повторного преобразования')
    if automatic:
        if physical:
            values['phase_rms_rad'] = scenario['phase']
            values['doppler_hz'] = scenario['doppler']
            values['distance_km'] = request.layer_height_km
        if scattering=='clarke':
            target = max(40,math.ceil(20*values['doppler_hz']))
            values['sample_rate_hz'] = min(160,target)
            if target>160:
                notes.append('Fs ограничена 160 Гц. Запас 20fD уменьшен; условие Найквиста Fs>2fD сохраняется.')
            values['waves'] = 128
            values['paths'] = scenario['paths'] if values['mode']=='wide' else 1
            values['delay_span_us'] = scenario['delay'] if values['mode']=='wide' else 0.
            record('doppler_hz','Максимальный Доплер fD',values['doppler_hz'],'Гц','scenario' if physical else 'input','Из выбранного сценария' if physical else 'Ключевой независимый параметр')
            record('sample_rate_hz','Частота отсчётов Fs',values['sample_rate_hz'],'Гц','numerical','min(160, max(40, ceil(20fD))); всегда Fs>2fD')
            record('waves','Гармоник на луч',values['waves'],'','scenario','N=128: инженерное приближение изотропного ансамбля')
            record('paths','Разрешаемых лучей',values['paths'],'','scenario','В узкополосном режиме 1; в широкополосном — выбранный профиль')
            record('delay_span_us','Размах задержек',values['delay_span_us'],'мкс','scenario','Заданный профиль рассеяния; не определяется из TEC')
        if scattering=='screen':
            values['velocity_m_s'] = scenario['velocity']
            fc = values['carrier_mhz']*1e6
            fresnel = math.sqrt(C/fc*values['distance_km']*1000/(2*math.pi))
            scale = fresnel if fresnel>0 else 100.
            if fresnel==0:
                notes.append('При z=0 rF=0; для сетки принят сценарный опорный масштаб 100 м. Распространение отсутствует.')
            values['outer_scale_m'] = max(50., min(5000.,scenario['alpha']*scale))
            values['inner_scale_m'] = max(5., min(500.,scenario['beta']*scale))
            values['spectral_index'] = 11/3
            spacing = min(values['inner_scale_m']/2,scale/3,100.)
            if spacing<2:
                raise ValueError('Требуемый шаг экрана меньше поддерживаемых 2 м. Измените частоту/дистанцию или используйте ручной режим.')
            values['spacing_m'] = spacing
            size_needed = max(4*values['outer_scale_m']/spacing, values['velocity_m_s']*values['duration_s']/(.8*spacing),128)
            size = 2**math.ceil(math.log2(size_needed))
            if size>256:
                max_duration = .8*256*spacing/max(values['velocity_m_s'],1e-30)
                raise ValueError(f'Для этих условий требуется сетка {size}×{size}; предел 256×256. Уменьшите длительность до {max_duration:.2f} с или выберите другой сценарий/геометрию.')
            values['screen_size'] = size
            values['sample_rate_hz'] = min(160,max(40, math.ceil(4*values['velocity_m_s']/spacing)))
            record('phase_rms_rad','RMS фазы при 1575,42 МГц',values['phase_rms_rad'],'рад','scenario' if physical else 'input','Из выбранного сценария' if physical else 'Ключевой параметр силы фазовых флуктуаций')
            record('fresnel_m','Масштаб Френеля rF',fresnel,'м','formula','rF=√(λz/(2π)); конвенция циклов/м')
            record('outer_scale_m','Внешний масштаб L₀',values['outer_scale_m'],'м','scenario','L₀=4rF (с учётом поддерживаемых границ); допущение, не вывод из TEC')
            record('inner_scale_m','Внутренний масштаб l₀',values['inner_scale_m'],'м','scenario','l₀=0,4rF (с учётом поддерживаемых границ); допущение')
            record('spectral_index','Показатель спектра p',values['spectral_index'],'','scenario','p=11/3 для выбранного спектрального сценария')
            record('velocity_m_s','Скорость сканирования',values['velocity_m_s'],'м/с','scenario','Эффективная скорость замороженного поля; из профиля')
            record('spacing_m','Шаг сетки Δx',spacing,'м','numerical','Δx=min(l₀/2, rF/3, 100 м); при z=0 используется опорный масштаб')
            record('screen_size','Размер сетки',size,'× '+str(size),'numerical','Минимальная 2^m ≥128: NΔx≥4L₀ и vT≤0,8NΔx')
            record('sample_rate_hz','Частота отсчётов Fs',values['sample_rate_hz'],'Гц','numerical','min(160, max(40, ceil(4v/Δx)))')
        if scattering=='none':
            values['sample_rate_hz']=40
            record('sample_rate_hz','Частота отсчётов Fs',40,'Гц','numerical','Диагностическая сетка медленного слоя ПЭС')
    else:
        record('manual','Параметры модели','Ручной ввод','','input','Физические и численные параметры задаются пользователем')
    cfg = ChannelLabConfig.model_validate(values)
    fc=cfg.carrier_mhz*1e6
    if physical:
        path=40.3e16*cfg.stec_tecu/fc**2
        record('path_m','Дополнительный путь ΔLg',path,'м','formula','40,3·STEC·10¹⁶/fc²')
        record('delay_ns','Групповая задержка τg',path/C*1e9,'нс','formula','τg=ΔLg/c')
        record('phase_rad','Фазовое опережение',2*math.pi*40.3e16*cfg.stec_tecu/(C*fc),'рад','formula','φadv=+2π·40,3·STEC·10¹⁶/(cfc)')
        record('iono_doppler_hz','Частотная поправка по ПЭС',40.3e16*cfg.tec_rate_tecu_s/(C*fc),'Гц','formula','fion=40,3·10¹⁶/(cfc)·dSTEC/dt; отдельно от fD рассеяния')
    record('samples','Временных отсчётов',int(cfg.duration_s*cfg.sample_rate_hz),'','numerical','floor(T·Fs)')
    notes.append('Средний TEC не определяет RMS флуктуаций, fD и пространственные масштабы. Независимые параметры берутся из явно выбранного сценария.')
    return dict(config=cfg.model_dump(),parameters=rows,notes=notes,setup=dict(setup_mode=request.setup_mode,scenario=request.scenario,tec_kind=request.tec_kind,elevation_deg=request.elevation_deg,layer_height_km=request.layer_height_km))
