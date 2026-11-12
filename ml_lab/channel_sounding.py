"""Common sources, independent training and linear/circular frozen channels."""
import numpy as np
from ml_lab.radio_simulator import constellation, ber_metrics, iq


def sound_channel(response, frequency_hz, snapshot_s, *, seed, snr_db, noise_enabled, signal_type="ofdm", propagation="circular", cp_samples=32):
    """response is [frequency, snapshot], on a centered uniform DFT grid."""
    h = np.asarray(response).T
    n = h.shape[1]
    centered_fir = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(h, axes=1), axis=1), axes=1)
    delay_bins = np.arange(n)-n//2
    support = np.any(abs(centered_fir) > np.max(abs(centered_fir))*1e-12, axis=0)
    latency = max(0, -int(delay_bins[support].min())) if propagation == 'linear' and support.any() else 0
    # Center the finite-band impulse in a causal FIR. This introduces a known
    # numerical latency, retained in H and explicitly reported to the receiver.
    if latency:
        bins = np.arange(n)-n//2
        h = h*np.exp(-2j*np.pi*bins*latency/n)[None, :]
    rng = np.random.default_rng(seed)
    points, labels = constellation('QPSK')
    training = points[rng.integers(0, 4, h.shape)]
    indices = rng.integers(0, 4, h.shape)
    payload = points[indices]
    source_indices = indices
    if signal_type == 'qpsk':
        payload = np.fft.fftshift(np.fft.fft(points[indices], axis=1)/np.sqrt(n), axes=1)
    elif signal_type == 'gnss':
        # Seeded repeated spreading code with one independent navigation bit per
        # block. This is a GNSS-like test waveform, not GPS C/A or a receiver.
        code = rng.choice([-1., 1.], n)
        navigation = rng.integers(0, 2, (len(h), 1))
        chips = code[None, :]*(2*navigation-1)
        source_indices = ((chips+1)/2).astype(int)
        payload = np.fft.fftshift(np.fft.fft(chips, axis=1)/np.sqrt(n), axes=1)
    elif signal_type == 'lfm':
        u = np.arange(n)/n
        chirp = np.exp(1j*np.pi*n*(u*u-u))
        payload = np.tile(np.fft.fftshift(np.fft.fft(chirp)/np.sqrt(n)), (len(h), 1))
    nv = 10**(-snr_db/10) if noise_enabled else 0.

    def waveform(grid):
        return np.fft.ifft(np.fft.ifftshift(grid, axes=1), axis=1)*np.sqrt(n)

    # In linear mode training and payload form one contiguous stream for each
    # frozen snapshot. Filter state is preserved across the two OFDM blocks.
    def receive_pair(first, second):
        a, b = waveform(first), waveform(second)
        def prefix(w):
            return np.concatenate([w[:, -cp_samples:], w], axis=1) if cp_samples else w
        blocks = np.stack([prefix(a), prefix(b)], axis=1)
        stream = blocks.reshape(len(h), -1)
        fir = np.fft.ifft(np.fft.ifftshift(h, axes=1), axis=1)
        clean = np.array([np.convolve(row, tap)[:len(row)] for row, tap in zip(stream, fir)])
        noise = np.sqrt(nv/2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))
        useful = (clean+noise).reshape(len(h), 2, n+cp_samples)[:, :, cp_samples:]
        grids = np.fft.fftshift(np.fft.fft(useful, axis=2)/np.sqrt(n), axes=2)
        return grids[:, 0], useful[:, 1], grids[:, 1]

    def receive(grid):
        clean = waveform(h*grid)
        noise = np.sqrt(nv/2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))
        wave = clean+noise
        return wave, np.fft.fftshift(np.fft.fft(wave, axis=1)/np.sqrt(n), axes=1)

    if propagation == 'linear':
        training_rx, rx_wave, received = receive_pair(training, payload)
    else:
        _, training_rx = receive(training)
        rx_wave, received = receive(payload)
    estimated = training_rx/training
    equalized = received*np.conj(estimated)/np.maximum(abs(estimated)**2+nv, 1e-14)
    ideal = received*np.conj(h)/np.maximum(abs(h)**2+nv, 1e-14)
    true_cir = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(h, axes=1), axis=1), axes=1)
    estimated_cir = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(estimated, axes=1), axis=1), axes=1)
    fs = n*(frequency_hz[1]-frequency_hz[0])
    tx_wave = waveform(payload)
    def coherence(grid):
        correlation = []
        offsets = np.arange(n//2+1)*(frequency_hz[1]-frequency_hz[0])
        for lag in range(len(offsets)):
            a, b = grid[:, :n-lag], grid[:, lag:]
            denominator = np.sqrt(np.mean(abs(a)**2)*np.mean(abs(b)**2))
            correlation.append(float(abs(np.mean(b*a.conj()))/denominator) if denominator > 1e-30 else None)
        crossing = next((i for i in range(1, len(offsets)) if correlation[i] is not None and correlation[i-1] is not None and correlation[i]<.5<=correlation[i-1]), None)
        bc = None
        if crossing is not None:
            a, b = correlation[crossing-1:crossing+1]
            bc = float(offsets[crossing-1]+(a-.5)/(a-b)*(offsets[crossing]-offsets[crossing-1]))
        return dict(offset_hz=offsets.tolist(), correlation=correlation, bc_hz=bc,
                    method='First |R|=0.5 crossing of frequency-pair correlation across sounding snapshots; finite-band estimate, noise not subtracted')

    def characteristics(grid, cir):
        # Plot residual CIR on the physical relative-delay axis, excluding the
        # common numerical shift used solely to make the finite-band FIR causal.
        if latency:
            grid = grid*np.exp(2j*np.pi*bins*latency/n)[None, :]
            cir = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(grid, axes=1), axis=1), axes=1)
        return dict(coherence=coherence(grid), gain_db=(20*np.log10(np.maximum(abs(grid[0]), 1e-15))).tolist(),
                    phase_rad=np.unwrap(np.angle(grid[0])).tolist(), cir_power=(abs(cir[0])**2).tolist())
    def score(grid):
        if signal_type == 'ofdm':
            return ber_metrics(indices.ravel(), grid.ravel(), points, labels)
        samples = waveform(grid)
        if signal_type == 'qpsk':
            return ber_metrics(source_indices.ravel(), samples.ravel(), points, labels)
        if signal_type == 'gnss':
            return ber_metrics(source_indices.ravel(), samples.ravel(), np.array([-1, 1], complex), np.array([[0], [1]]))
        return dict(ber=None, errors=None, bits=0, interval95=None,
                    evm_percent=float(100*np.sqrt(np.mean(abs(samples-tx_wave)**2)/np.mean(abs(tx_wave)**2))))
    fir_power = abs(np.fft.ifft(np.fft.ifftshift(h, axes=1), axis=1))**2
    outside_cp = float(fir_power[:, cp_samples+1:].sum()/max(float(fir_power.sum()), 1e-30))
    return dict(cp_tail_energy_fraction=outside_cp if propagation=='linear' else None, generator=signal_type, propagation=propagation, cp_samples=cp_samples if propagation=='linear' else None,
                numerical_latency_samples=latency, ber_basis='chip decisions' if signal_type=='gnss' else 'payload bits' if signal_type!='lfm' else 'not applicable',  snapshots_s=np.asarray(snapshot_s).tolist(),
                sample_rate_hz=float(fs), block_samples=n, noise_variance=nv,
                assumptions='Frozen snapshots; no intra-block evolution/ICI. Linear mode preserves training-to-payload filter memory and actual CP; circular mode assumes sufficient guard. Full-band training independent of payload.',
                waveform=dict(time_s=(np.arange(n)/fs).tolist(), tx=iq(tx_wave[0], n), rx=iq(rx_wave[0], n)),
                raw=score(received), receiver=score(equalized), ideal_reference=score(ideal),
                truth=characteristics(h, true_cir), estimated=characteristics(estimated, estimated_cir),
                verification=dict(channel_nmse=float(np.sum(abs(estimated-h)**2)/max(float(np.sum(abs(h)**2)), 1e-30)),
                                  parseval_error=float(abs(np.mean(abs(tx_wave)**2)-np.mean(abs(payload)**2)))))
