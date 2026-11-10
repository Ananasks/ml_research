"""First-order cold plasma; ITU-R P.531-16 §§4.4–4.6.
Convention Re{x exp(+j2πft)}: positive transfer phase advance, negative phase range.
"""
import numpy as np

C = 299792458.0
ITU_SOURCE = 'https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.531-16-202509-I!!PDF-E.pdf'


def mapping(elevation_deg, height_km=350.):
    return 1 / np.sqrt(1-(6371*np.cos(np.deg2rad(elevation_deg))/(6371+height_km))**2)


def physical(stec_tecu, frequency_hz, bandwidth_hz=0):
    stec = np.asarray(stec_tecu)*1e16
    f = frequency_hz
    path = 40.3*stec/f**2
    # User-specified dispersive bandwidth criterion (80.5 is its rounded constant).
    bandwidth = np.full_like(stec, np.inf, dtype=float)
    np.divide(C*f**3, 80.5*np.pi*stec, out=bandwidth, where=stec > 0)
    bandwidth = np.sqrt(bandwidth)
    return {'coherence_bandwidth_mhz': bandwidth/1e6,
            'delay_dispersion_s2': -80.5*stec/(2*np.pi*C*f**3), 'stec_tecu': np.asarray(stec_tecu), 'path_m': path, 'delay_ns': path/C*1e9,
            'phase_rad': 2*np.pi*40.3*stec/(C*f),
            'dispersion_ns': 40.3*stec/C*((f-bandwidth_hz/2)**-2-(f+bandwidth_hz/2)**-2)*1e9}


def transfer(stec_tecu, frequency_hz):
    return np.exp(2j*np.pi*40.3*np.asarray(stec_tecu)*1e16/(C*frequency_hz))
