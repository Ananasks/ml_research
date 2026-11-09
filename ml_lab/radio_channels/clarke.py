"""Statistical isotropic-scattering benchmark; not an ionospheric law."""
import numpy as np


def clarke(t, fd, seed, waves=128):
    """Random angles/phases SoS; E|g|²=1, ensemble R=J0(2πfdτ).
    No per-realization normalization: preserves deep/slow fades.
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, waves)
    phases = rng.uniform(-np.pi, np.pi, waves)
    t = np.asarray(t)
    return np.exp(1j*(2*np.pi*fd*t[..., None]*np.cos(angles)+phases)).sum(axis=-1)/np.sqrt(waves)

