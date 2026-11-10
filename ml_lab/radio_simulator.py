"""Reproducible complex-baseband radio experiments; see docs/radio_simulator.md.
Independent implementation of published equations (no vendored third-party code).
Independent TEC physical layer and Clarke statistical reference.
"""
from __future__ import annotations
from typing import Literal
import numpy as np
from scipy.signal import welch
from scipy.special import j0, erfc
from pydantic import BaseModel, Field, model_validator

from ml_lab.radio_channels import clarke, MODEL_REGISTRY
from ml_lab.radio_channels.tec import transfer, C


class RadioConfig(BaseModel):
    mode: Literal['narrow', 'wide'] = 'narrow'
    model: Literal['clarke','tec','combined'] = 'clarke'
    carrier_mhz: float = Field(1575.42,ge=100,le=12000)
    tec_stec_tecu: float = Field(0,ge=0,le=2000)
    tec_rate_tecu_s: float = Field(0,ge=-100,le=100)
    compensation_stec_tecu: float | None = Field(None,ge=0,le=2000)
    compensation_rate_tecu_s: float = Field(0,ge=-100,le=100)
    channel_estimation: Literal['ideal','pilot','forecast'] = 'ideal'
    realizations: int = Field(1,ge=1,le=10)
    source_context: dict = Field(default_factory=dict)
    modulation: Literal['BPSK', 'QPSK', '16-QAM'] = 'QPSK'
    doppler_hz: float = Field(1, ge=0, le=10)
    clarke_waves: int = Field(128, ge=16, le=256)
    diagnostic_duration_s: float = Field(60, ge=5, le=120)
    diagnostic_rate_hz: int = Field(40, ge=40, le=160)
    offset_hz: float = Field(0, ge=-20, le=20)
    snr_db: float = Field(15, ge=-5, le=40)
    rate_hz: float = Field(10000, ge=1000, le=20000000)
    nfft: Literal[128,256,512,1024,2048] = 256
    cp_divisor: Literal[4,8,16] = 4
    symbols: int = Field(8192, ge=2048, le=32768)
    seed: int = Field(42, ge=0, le=2147483647)
    fading_enabled: bool = True
    noise_enabled: bool = True
    echo_enabled: bool = True
    echo_delay_us: float = Field(2, ge=0, le=500)
    echo_db: float = Field(-6, ge=-40, le=0)
    equalizer: Literal['zf', 'mmse'] = 'mmse'
    sweep: bool = True

    @model_validator(mode='after')
    def check_band(self):
        if self.rate_hz >= 2*self.carrier_mhz*1e6:
            raise ValueError('Полоса пересекает нулевую абсолютную частоту.')
        if self.channel_estimation == 'forecast' and self.compensation_stec_tecu is None:
            raise ValueError('Для компенсации по прогнозу выберите прогноз ПЭС на том же времени.')
        if self.mode == 'wide' and self.echo_enabled and self.echo_delay_us * 1e-6 * self.rate_hz > 2048:
            raise ValueError('Задержка эха превышает 2048 отсчётов; уменьшите полосу или задержку.')
        return self


def channel(config, t, seed):
    fading = clarke(t, config.doppler_hz, seed, waves=config.clarke_waves) if config.fading_enabled and config.model != 'tec' else np.ones_like(t, dtype=complex)
    plasma = transfer(np.maximum(0, config.tec_stec_tecu+config.tec_rate_tecu_s*t),config.carrier_mhz*1e6) if config.model!='clarke' else 1
    return fading * plasma * np.exp(2j*np.pi*config.offset_hz*t)


def constellation(modulation):
    if modulation == 'BPSK':
        return np.array([-1, 1], dtype=complex), np.array([[0], [1]])
    if modulation == 'QPSK':
        levels, bits = np.array([-1, 1])/np.sqrt(2), np.array([[0], [1]])
    else:
        levels, bits = np.array([-3, -1, 1, 3])/np.sqrt(10), np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    return np.array([a+1j*b for a in levels for b in levels]), np.array([np.r_[a, b] for a in bits for b in bits])


def decisions(samples, points):
    return np.abs(samples[..., None]-points).argmin(axis=-1)


def ber_metrics(tx_idx, samples, points, labels):
    rx_idx = decisions(samples, points)
    errors = int(np.count_nonzero(labels[tx_idx] != labels[rx_idx]))
    bits = int(tx_idx.size*labels.shape[1])
    p, z = errors/bits, 1.95996398454
    center = (p+z*z/(2*bits))/(1+z*z/bits)
    radius = z*np.sqrt(p*(1-p)/bits+z*z/(4*bits*bits))/(1+z*z/bits)
    return dict(ber=p, errors=errors, bits=bits, interval95=[max(0., center-radius), min(1., center+radius)],
                evm_percent=float(100*np.sqrt(np.mean(np.abs(samples-points[tx_idx])**2)/np.mean(np.abs(points[tx_idx])**2))))


def error_statistics(tx_idx, samples, points, labels):
    """Observed error runs and block errors, not a purported Blank model.

    A burst is strictly consecutive wrong bits; zero correct bits allowed within.
    Block length is 128 bits. The trailing incomplete block is excluded.
    """
    wrong = (labels[tx_idx] != labels[decisions(samples, points)]).ravel()
    padded = np.r_[False, wrong, False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    lengths = ends-starts
    values, counts = np.unique(lengths, return_counts=True)
    size = 128
    blocks = wrong[:len(wrong)//size*size].reshape(-1, size).sum(axis=1)
    centered = wrong.astype(float)-wrong.mean()
    var = float(np.mean(centered**2))
    lags = np.arange(65)
    acf = [float(np.mean(centered[:len(wrong)-lag]*centered[lag:])/var) if var > 0 else None for lag in lags]
    return {'burst_lengths': values.tolist(), 'burst_counts': counts.tolist(),
            'max_burst_bits': int(lengths.max()) if len(lengths) else 0,
            'burst_count': int(len(lengths)), 'block_bits': size,
            'block_count': int(len(blocks)), 'block_error_rate': float(np.mean(blocks > 0)),
            'error_lags_bits': lags.tolist(), 'error_acf': acf}


def iq(z, limit=1200):
    z = np.asarray(z).ravel()[:limit]
    return {'i': z.real.tolist(), 'q': z.imag.tolist()}


def spectrum(x, fs):
    f, p = welch(x, fs=fs, nperseg=min(512, len(x)), return_onesided=False, detrend=False)
    order = np.argsort(f)
    return {'f': f[order].tolist(), 'db': (10*np.log10(np.maximum(p[order], 1e-20))).tolist()}


def experiment(config, model=None, compact=False):
    if model:
        config = config.model_copy(update={'model': model})
    rng = np.random.default_rng(config.seed)
    points, labels = constellation(config.modulation)
    nfft, cp = config.nfft, config.nfft//config.cp_divisor
    count = config.symbols if config.mode == 'narrow' else max(1, config.symbols//nfft)*nfft
    indices = rng.integers(0, len(points), count)
    x = points[indices]
    fs = config.rate_hz
    noise_var = 10**(-config.snr_db/10) if config.noise_enabled else 0.
    taps, frequency_map = None, None
    if config.mode == 'narrow':
        t = np.arange(count)/fs
        h = channel(config, t, config.seed+101)
        clean = x*h
        tx_wave, clean_wave = x, clean
        coef = h
    else:
        blocks = count//nfft
        t = np.arange(blocks)*(nfft+cp)/fs
        # Block-frozen fading. CP protects discrete taps; actual linear filtering
        # includes previous OFDM blocks, including CP-shortfall interference.
        h = channel(config, t, config.seed+101)
        echo = config.echo_enabled
        delay = int(round(config.echo_delay_us*1e-6*fs)) if echo else 0
        power = 10**(config.echo_db/10) if echo else 0.
        a = np.sqrt(1/(1+power))
        b = np.sqrt(power/(1+power))*np.exp(1j*np.pi/3)
        impulse = np.zeros(delay+1, dtype=complex)
        impulse[0] = a
        if echo:
            impulse[delay] += b
        td = np.fft.ifft(x.reshape(blocks, nfft), axis=1)*np.sqrt(nfft)
        tx_wave = np.concatenate([td[:, -cp:], td], axis=1).ravel()
        delayed = np.convolve(tx_wave, impulse)[:len(tx_wave)].reshape(blocks, nfft+cp)
        nu = np.fft.fftfreq(nfft, 1/fs)
        tap_response = sum(value*np.exp(-2j*np.pi*nu*k/fs) for k, value in enumerate(impulse) if value != 0)
        base = np.fft.fft(delayed[:, cp:], axis=1)/np.sqrt(nfft)
        fc=config.carrier_mhz*1e6
        k=2*np.pi*40.3e16*np.maximum(0,config.tec_stec_tecu+config.tec_rate_tecu_s*t)/C
        dispersion=np.exp(1j*k[:,None]*(1/(fc+nu)[None,:]-1/fc+nu[None,:]/fc**2)) if config.model!='clarke' else np.ones((blocks,nfft),complex)
        clean_matrix = base*h[:, None]*dispersion
        coef = h[:, None]*tap_response[None, :]*dispersion
        clean = clean_matrix.ravel()
        # Spectrum of reconstructed received useful blocks; CP not included.
        clean_wave = (np.fft.ifft(clean_matrix, axis=1)*np.sqrt(nfft)).ravel()
        taps = {'delay_us': (np.arange(len(impulse))/fs*1e6).tolist(), 'power': np.abs(impulse).tolist(), 'delay_samples': delay, 'cp': cp}
        frequency_map = {'time': t.tolist(), 'frequency_mhz': (np.fft.fftshift(nu)/1e6).tolist(), 'gain_db': (20*np.log10(np.maximum(np.abs(np.fft.fftshift(coef, axes=1)), 1e-10))).tolist()}
    # Es/N0 at unit transmit power, never normalized to each faded realization.
    noise = (rng.normal(size=count)+1j*rng.normal(size=count))/np.sqrt(2)
    received = clean+np.sqrt(noise_var)*noise
    coeff = np.asarray(coef).ravel()
    pilot_mask = np.zeros(count,dtype=bool)
    if config.channel_estimation=='pilot':
        pilot_mask[::16]=True
    data_mask=~pilot_mask

    def estimate(noisy):
        if config.channel_estimation=='ideal':
            return coeff
        if config.channel_estimation=='pilot':
            if config.mode=='narrow':
                positions=np.flatnonzero(pilot_mask)
                least=noisy[positions]/x[positions]
                return np.interp(np.arange(count),positions,least.real)+1j*np.interp(np.arange(count),positions,least.imag)
            matrix=noisy.reshape(-1,nfft); known=x.reshape(-1,nfft)
            positions=np.arange(0,nfft,16)
            least=matrix[:,positions]/known[:,positions]
            return np.array([np.interp(np.arange(nfft),positions,row.real)+1j*np.interp(np.arange(nfft),positions,row.imag) for row in least]).ravel()
        # Isolate TEC forecast error: other channel components are known exactly.
        actual=np.maximum(0,config.tec_stec_tecu+config.tec_rate_tecu_s*t)
        predicted=np.maximum(0,config.compensation_stec_tecu+config.compensation_rate_tecu_s*t)
        if config.model=='clarke':
            return coeff
        if config.mode=='narrow':
            ratio=transfer(predicted-actual,config.carrier_mhz*1e6)
        else:
            fc=config.carrier_mhz*1e6
            delta=2*np.pi*40.3e16*(predicted-actual)/C
            ratio=np.exp(1j*delta[:,None]*(1/(fc+nu)[None,:]+nu[None,:]/fc**2)).ravel()
        return coeff*ratio

    def decode(noisy,nv):
        estimated=estimate(noisy)
        gain=np.conj(estimated)/np.maximum(np.abs(estimated)**2+(nv if config.equalizer=='mmse' else 0),1e-14)
        return noisy*gain

    equalized = decode(received,noise_var)
    raw_metrics = ber_metrics(indices[data_mask], received[data_mask], points, labels)
    metrics = ber_metrics(indices[data_mask], equalized[data_mask], points, labels)
    if compact:
        return {'metrics': metrics, 'raw_metrics': raw_metrics,
                'reference_energy': float(np.sum(np.abs(x[data_mask])**2)),
                'error_energy': float(np.sum(np.abs(equalized[data_mask]-x[data_mask])**2))}
    sweep = []
    if config.sweep:
        for snr in range(0, 31, 5):
            nv = 10**(-snr/10)
            noisy = clean+np.sqrt(nv)*noise
            decoded=decode(noisy,nv)
            score = ber_metrics(indices[data_mask], decoded[data_mask], points, labels)
            before=ber_metrics(indices[data_mask],noisy[data_mask],points,labels)
            sweep.append({'snr_db': snr, **score, 'raw_ber':before['ber'], 'raw_evm_percent':before['evm_percent']})
    if config.mode == 'wide':
        received_wave = (np.fft.ifft(received.reshape(-1, nfft), axis=1)*np.sqrt(nfft)).ravel()
    else:
        received_wave = received
    return dict(model=config.model, metrics=metrics, raw_metrics=raw_metrics,
                duration_s=float(count/fs if config.mode == 'narrow' else (count//nfft)*(nfft+cp)/fs),
                iq={'tx': iq(x), 'channel': iq(clean), 'noise': iq(received), 'receiver': iq(equalized[data_mask]), 'reference': iq(points)},
                pilot_symbols=int(pilot_mask.sum()), channel_estimation=config.channel_estimation,
                waveform={'tx_real':(td.ravel() if config.mode=='wide' else x).real[:2048].tolist(), 'time_s': (np.arange(min(2048, len(received_wave)))/fs).tolist(),
                          'real': received_wave.real[:2048].tolist(), 'imag': received_wave.imag[:2048].tolist()},
                error_statistics=error_statistics(indices[data_mask], equalized[data_mask], points, labels),
                spectrum={'tx': spectrum(tx_wave, fs), 'rx': spectrum(received_wave, fs)},
                sweep=sweep, taps=taps, response=frequency_map,
                rx_power=float(np.mean(np.abs(clean)**2)), noise_variance=noise_var,
                cp_sufficient=(taps is None or taps['delay_samples'] <= cp))


def simulate(config: RadioConfig):
    primary = experiment(config)
    from scipy.stats import t as student_t
    runs=[primary]
    for index in range(1,config.realizations):
        runs.append(experiment(config.model_copy(update={'seed':(config.seed+index)%2147483648,'sweep':False})))
    values=np.array([run['metrics']['ber'] for run in runs])
    radius=float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    ensemble={'realizations':len(values),'seeds':[(config.seed+i)%2147483648 for i in range(len(values))],
        'mean_ber':float(values.mean()),'interval95': [max(0.,float(values.mean())-radius),min(1.,float(values.mean())+radius)] if radius is not None else None,
        'method':'Student t over independent realization BER means; unavailable for one realization'}
    # A separate 60 s observation, NOT stretched symbol time; same process seed.
    dt = 1/config.diagnostic_rate_hz
    t = np.arange(round(config.diagnostic_duration_s*config.diagnostic_rate_hz))*dt
    g = channel(config, t, config.seed+101)
    intensity = np.abs(g)**2
    fading = clarke(t, config.doppler_hz, config.seed+101, waves=config.clarke_waves)
    centered = fading-fading.mean()
    variance = float(np.mean(np.abs(centered)**2))
    lags = np.arange(min(201,len(t)//2))
    acf = [float(np.real(np.mean(centered[:len(t)-lag]*np.conj(centered[lag:])))/variance) if variance > 1e-20 else None for lag in lags]
    channel_centered=g-g.mean()
    channel_variance=float(np.mean(np.abs(channel_centered)**2))
    channel_acf=[float(np.real(np.mean(channel_centered[:len(t)-lag]*np.conj(channel_centered[lag:])))/channel_variance) if channel_variance>1e-20 else None for lag in lags]
    paired=None
    if config.source_context.get('input_mode')=='compare' and config.compensation_stec_tecu is not None and config.model!='clarke':
        paired=experiment(config.model_copy(update={'tec_stec_tecu':config.compensation_stec_tecu,'tec_rate_tecu_s':config.compensation_rate_tecu_s}))
    metrics = primary['metrics']
    warnings = [
        'Кларк — эталон изотропного рассеяния, не физическая модель ионосферных сцинтилляций.',
        f'Оценка канала: {config.channel_estimation}. Пилоты 1/16 исключаются из BER/EVM; forecast изолирует ошибку ПЭС при известном остальном канале.',
        'SNR означает Es/N0 на единичной мощности передатчика; частота дискретизации относится к комплексной огибающей.',
        'Доверительный интервал BER: приближение Wilson для независимых битов; при коррелированных замираниях неопределённость больше.',
    ]
    if config.mode == 'wide':
        warnings += [f'OFDM: {config.nfft} поднесущих, CP={config.nfft//config.cp_divisor}; замирания Кларка фиксированы внутри блока. ICI от изменений внутри блока не моделируется.',
                     'Дополнительный луч — заданный пользователем сценарий испытаний. Задержка округляется до ближайшего отсчёта.']
        if not primary['cp_sufficient']:
            warnings += ['CP короче задержки эха: линейная свёртка создаёт ISI, одноотводный эквалайзер не устраняет её.']
    if config.mode=='wide' and (config.doppler_hz+abs(config.offset_hz))*(config.nfft+config.nfft//config.cp_divisor)/config.rate_hz > .1:
        warnings += ['Канал меняется существенно за OFDM-блок; приближение постоянного внутри блока H может быть неточным.']
    if config.model!='clarke':
        warnings += ['ПЭС линейно продолжается на длительность пакета от выбранного времени; общая задержка компенсирована, дисперсия OFDM применяется циклически без межблочной памяти.']
    if metrics['errors'] < 100:
        warnings += ['Мало битовых ошибок: увеличьте число символов и повторите с другими seed; BER=0 не означает нулевую вероятность ошибки.']
    if config.doppler_hz*config.diagnostic_duration_s < 10:
        warnings += [f'За {config.diagnostic_duration_s:g} с мало периодов замирания; эмпирические ACF и S4 могут сильно отличаться от ансамблевых значений.']
    from ml_lab.clarke_statistics import fading_statistics
    fading_stats = fading_statistics(t, np.abs(fading), config.doppler_hz)
    kbits = constellation(config.modulation)[1].shape[1]
    snrs = np.arange(0, 31, 5)
    ebn0 = 10**(snrs/10)/kbits
    # Exact Gray BPSK/QPSK reference; 16-QAM is labelled high-SNR approximation.
    awgn = .5*erfc(np.sqrt(ebn0)) if kbits < 4 else .375*erfc(np.sqrt(.4*ebn0))
    return {'config': config.model_dump(), 'primary': primary, 'predicted_channel':paired, 'ensemble':ensemble, 'warnings': warnings, 'models': MODEL_REGISTRY,
            'clarke_statistics': fading_stats,
            'channel_statistics': fading_statistics(t, np.abs(g), config.doppler_hz if config.model!='tec' and config.fading_enabled else 0),
            'diagnostics': {'channel_acf':channel_acf, 'channel_psd':spectrum(g,1/dt), 'time': t.tolist(), 'amplitude': np.abs(g).tolist(), 'phase': np.unwrap(np.angle(g)).tolist(),
                'real': g.real.tolist(), 'imag': g.imag.tolist(),
                's4': float(np.std(intensity)/np.mean(intensity)), 'mean_power': float(intensity.mean()),
                'lag_s': (lags*dt).tolist(), 'acf': acf, 'acf_theory': j0(2*np.pi*config.doppler_hz*lags*dt).tolist(),
                'psd': spectrum(fading, 1/dt)},
            'awgn': {'snr_db': snrs.tolist(), 'ber': awgn.tolist(), 'approximation': kbits == 4}}
