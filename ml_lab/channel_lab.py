"""Independent bounded SoS and single thin-screen Fresnel experiments.

The screen is a finite periodic Gaussian spectral synthesis, not an implementation
of the complete Rino model or all of ITU-R P.531. See docs/channel_lab.md.
"""
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.signal import correlate, welch
from scipy.special import j0
from ml_lab.radio_channels.clarke import clarke

C = 299792458.0
REFERENCE_HZ = 1575.42e6


class ChannelLabConfig(BaseModel):
    model: Literal['clarke', 'screen', 'physical'] = 'clarke'
    mode: Literal['narrow', 'wide'] = 'narrow'
    carrier_mhz: float = Field(1575.42, ge=100, le=12000)
    bandwidth_mhz: float = Field(1, ge=.01, le=20)
    duration_s: float = Field(30, ge=2, le=120)
    sample_rate_hz: int = Field(40, ge=10, le=160)
    seed: int = Field(42, ge=0, le=2147483647)
    doppler_hz: float = Field(1, ge=0, le=30)
    waves: int = Field(128, ge=16, le=256)
    paths: int = Field(4, ge=1, le=16)
    delay_span_us: float = Field(2, ge=0, le=50)
    screen_size: Literal[128, 256] = 128
    spacing_m: float = Field(20, ge=2, le=100)
    distance_km: float = Field(350, ge=0, le=1000)
    velocity_m_s: float = Field(40, ge=0, le=500)
    phase_rms_rad: float = Field(.5, ge=0, le=3)
    outer_scale_m: float = Field(500, ge=50, le=5000)
    inner_scale_m: float = Field(40, ge=5, le=500)
    spectral_index: float = Field(11/3, ge=2.1, le=4.5)
    scattering: Literal['none', 'clarke', 'screen'] = 'screen'
    stec_tecu: float = Field(10, ge=0, le=2000)
    tec_rate_tecu_s: float = Field(0, ge=-10, le=10)

    @model_validator(mode='after')
    def sampling(self):
        if self.sample_rate_hz <= 2*self.doppler_hz and (self.model == 'clarke' or self.model == 'physical' and self.scattering == 'clarke'):
            raise ValueError('Для Кларка требуется Fs > 2 fD.')
        if self.uses_screen and self.duration_s*self.velocity_m_s > .8*self.screen_size*self.spacing_m:
            raise ValueError('Траектория превышает 80% экрана. Уменьшите T или скорость либо увеличьте сетку/шаг.')
        if self.inner_scale_m >= self.outer_scale_m:
            raise ValueError('Внутренний масштаб должен быть меньше внешнего.')
        if min(self.stec_tecu, self.stec_tecu+self.tec_rate_tecu_s*self.duration_s) < 0:
            raise ValueError('Линейное продолжение STEC становится отрицательным; уменьшите длительность или производную.')
        return self

    @property
    def uses_screen(self):
        return self.model == 'screen' or self.model == 'physical' and self.scattering == 'screen'


def phase_screen(cfg):
    """Real Gaussian field with prescribed finite-grid spatial RMS at reference f.

    q is spatial frequency in cycles/m; PSD ∝(q²+L0⁻²)^(-p/2)
    exp[-(q l0)²]. White real noise guarantees Hermitian symmetry.
    """
    n = cfg.screen_size
    q = np.fft.fftfreq(n, cfg.spacing_m)
    q2 = q[:, None]**2 + q[None, :]**2
    psd = (q2+cfg.outer_scale_m**-2)**(-cfg.spectral_index/2)*np.exp(-q2*cfg.inner_scale_m**2)
    psd[0, 0] = 0
    white = np.random.default_rng(cfg.seed).normal(size=(n, n))
    phi = np.fft.ifft2(np.fft.fft2(white)*np.sqrt(psd)).real
    phi -= phi.mean()
    phi *= cfg.phase_rms_rad/max(float(phi.std()), 1e-30)
    return phi


def propagate(phi, frequency_hz, cfg):
    q = np.fft.fftfreq(cfg.screen_size, cfg.spacing_m)
    kernel = np.exp(1j*np.pi*(C/frequency_hz)*cfg.distance_km*1000*(q[:, None]**2+q[None, :]**2))
    return np.fft.ifft2(np.fft.fft2(np.exp(1j*phi*REFERENCE_HZ/frequency_hz))*kernel)


def sample_track(field, t, cfg):
    # One finite frozen-flow traversal, no repeated wrapping of the screen.
    x = .1*cfg.screen_size + cfg.velocity_m_s*t/cfg.spacing_m
    x0 = np.floor(x).astype(int)
    weight = x-x0
    y = cfg.screen_size//2
    return field[y, x0]*(1-weight)+field[y, np.minimum(x0+1, cfg.screen_size-1)]*weight


def run_channel_lab(cfg: ChannelLabConfig):
    fc, band = cfg.carrier_mhz*1e6, cfg.bandwidth_mhz*1e6
    t = np.arange(int(cfg.duration_s*cfg.sample_rate_hz))/cfg.sample_rate_hz
    physical = cfg.model == 'physical'
    scattering = cfg.scattering if physical else cfg.model
    warnings = []
    phi = field = None
    taps = None
    if scattering == 'screen':
        phi = phase_screen(cfg)
        field = propagate(phi, fc, cfg)
        h = np.exp(1j*sample_track(phi, t, cfg)*REFERENCE_HZ/fc) if cfg.distance_km == 0 else sample_track(field, t, cfg)
        fresnel = np.sqrt(C/fc*cfg.distance_km*1000/(2*np.pi))
        if fresnel and cfg.spacing_m > fresnel/3:
            warnings.append('Шаг экрана крупнее rF/3: уточните сетку и сравните результаты.')
        if cfg.screen_size*cfg.spacing_m < 4*cfg.outer_scale_m:
            warnings.append('Размер экрана меньше четырёх внешних масштабов: крупные неоднородности ограничены сеткой.')
        if cfg.spacing_m > cfg.inner_scale_m/2:
            warnings.append('Внутренний масштаб разрешён менее чем двумя ячейками. Уменьшите шаг сетки.')
        warnings.append('Один тонкий периодический экран; края и конечная сетка влияют на результат. Проверяйте сходимость по шагу и размеру.')
    elif scattering == 'clarke':
        count = cfg.paths if cfg.mode == 'wide' else 1
        delays = np.linspace(0, cfg.delay_span_us*1e-6, count)
        power = np.exp(-np.linspace(0, 3, count)); power /= power.sum()
        gains = np.array([clarke(t, cfg.doppler_hz, cfg.seed+101+i*997, cfg.waves)*np.sqrt(p) for i,p in enumerate(power)])
        h = gains.sum(axis=0)
        taps = (delays, power, gains)
        warnings.append('Кларк — эталон изотропного рассеяния. Число лучей, задержки и fD заданы сценарием, не восстановлены из TEC.')
    else:
        h = np.ones(len(t), dtype=complex)
    s = cfg.stec_tecu+cfg.tec_rate_tecu_s*t if physical else np.zeros_like(t)
    k = 2*np.pi*40.3e16*s/C
    h *= np.exp(1j*k/fc)
    # Symmetric finite-band frequency response, at the initial state. For screen
    # correlations use many positions, rather than correlating one scalar pair.
    nu = (np.arange(129)-64)*band/129
    frequency = fc+nu
    track_indices = np.linspace(0, len(t)-1, min(128,len(t))).astype(int)
    if cfg.mode == 'narrow':
        response = np.full(len(nu), h[0], dtype=complex)
        matrix = np.tile(h[track_indices], (len(nu),1))
    else:
        if scattering == 'screen':
            matrix = np.array([np.exp(1j*sample_track(phi,t[track_indices],cfg)*REFERENCE_HZ/f) if cfg.distance_km == 0 else sample_track(propagate(phi,f,cfg),t[track_indices],cfg) for f in frequency])
        elif taps is not None:
            delays, power, gains = taps
            matrix = np.exp(-2j*np.pi*nu[:,None]*delays)@gains[:,track_indices]
        else:
            matrix = np.ones((len(nu),len(track_indices)),dtype=complex)
        if physical:
            # Remove common carrier phase and linear group delay for residual CIR;
            # show absolute physical delay and phase separately.
            kk = k[track_indices]
            matrix *= np.exp(1j*kk[None,:]*(1/frequency[:,None]-1/fc+nu[:,None]/fc**2))
        response = matrix[:,0]
    correlations = []
    for lag in range(len(nu)//2+1):
        a = matrix[:len(nu)-lag]; b = matrix[lag:]
        den = np.sqrt(np.mean(np.abs(a)**2)*np.mean(np.abs(b)**2))
        correlations.append(float(abs(np.mean(b*a.conj()))/den) if den > 1e-30 else None)
    if taps is not None:
        delays,power,_ = taps
        offsets = np.arange(len(correlations))*(nu[1]-nu[0])
        correlations = np.abs(np.exp(-2j*np.pi*offsets[:,None]*delays)@power).tolist() if cfg.mode=='wide' else [1.]*len(offsets)
    offsets = np.arange(len(correlations))*(nu[1]-nu[0])
    crossing = next((i for i,v in enumerate(correlations) if v is not None and v<.5),None)
    bc = None
    if crossing:
        a,b = correlations[crossing-1:crossing+1]
        bc = float(offsets[crossing-1]+(a-.5)/(a-b)*(offsets[crossing]-offsets[crossing-1]))
    impulse = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(response)))
    delay_axis = np.fft.fftshift(np.fft.fftfreq(len(nu),d=nu[1]-nu[0]))*1e6
    centered = h-h.mean()
    variance = float(np.mean(abs(centered)**2))
    nlags = min(len(t)//2,512)
    if variance > 1e-20:
        ac = correlate(centered,centered,mode='full',method='fft')[len(t)-1:len(t)-1+nlags]
        ac = (ac/(np.arange(len(t),len(t)-nlags,-1)*variance)).real.tolist()
    else:
        ac = [None]*nlags
    f,p = welch(h,fs=cfg.sample_rate_hz,nperseg=min(512,len(h)),return_onesided=False,detrend=False)
    order = np.argsort(f)
    intensity = abs(h)**2
    lag_s = np.arange(nlags)/cfg.sample_rate_hz
    theoretical = j0(2*np.pi*cfg.doppler_hz*lag_s).tolist() if scattering=='clarke' else None
    df = np.linspace(-.999*cfg.doppler_hz,.999*cfg.doppler_hz,501) if cfg.doppler_hz else np.array([])
    theory_psd = 1/(np.pi*cfg.doppler_hz*np.sqrt(1-(df/cfg.doppler_hz)**2)) if len(df) else []
    if scattering=='screen' and cfg.velocity_m_s == 0:
        warnings.append('Нулевая скорость: приёмник неподвижен в замороженном поле, временных сцинтилляций нет.')
    warnings.append('Bc: первое пересечение |R|=0,5 в исследованной полосе. Отсутствие пересечения не означает измеренную бесконечную Bc.')
    return dict(config=cfg.model_dump(), time_s=t.tolist(), real=h.real.tolist(), imag=h.imag.tolist(), amplitude=abs(h).tolist(), phase=np.unwrap(np.angle(h)).tolist(),
                lag_s=lag_s.tolist(),acf=ac,acf_theory=theoretical,
                psd=dict(f=f[order].tolist(),density=p[order].real.tolist()),theory_psd=dict(f=df.tolist(),density=np.asarray(theory_psd).tolist()),
                frequency_mhz=(nu/1e6).tolist(),gain_db=(20*np.log10(np.maximum(abs(response),1e-15))).tolist(),response_phase=np.unwrap(np.angle(response)).tolist(),
                delay_us=delay_axis.tolist(),cir_power=(abs(impulse)**2).tolist(),correlation=dict(mhz=(offsets/1e6).tolist(),value=correlations),
                physical=dict(stec_tecu=s.tolist(),path_m=(40.3e16*s/fc**2).tolist(),delay_ns=(40.3e16*s/(C*fc**2)*1e9).tolist(),phase_rad=(k/fc).tolist(),doppler_hz=40.3e16*cfg.tec_rate_tecu_s/(C*fc) if physical else 0),
                screen=None if phi is None else dict(structure=dict(rho_m=(np.arange(1,cfg.screen_size//4)*cfg.spacing_m).tolist(),value=[float(np.mean(((phi[:,i:]-phi[:,:-i])*REFERENCE_HZ/fc)**2)) for i in range(1,cfg.screen_size//4)]),x_m=(np.arange(cfg.screen_size)*cfg.spacing_m).tolist(),phase_rad=(phi*REFERENCE_HZ/fc).tolist(),intensity=(abs(field)**2).tolist(),mean_intensity=float(np.mean(abs(field)**2))),
                taps=None if taps is None else dict(delay_us=(taps[0]*1e6).tolist(),power=taps[1].tolist()),
                metrics=dict(mean_power=float(intensity.mean()),s4=float(intensity.std()/max(intensity.mean(),1e-30)),phase_std_rad=float(np.std(np.unwrap(np.angle(h)))),bc_mhz=None if bc is None else bc/1e6),warnings=warnings)
