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
    return {'stec_tecu': np.asarray(stec_tecu), 'path_m': path, 'delay_ns': path/C*1e9,
            'phase_rad': 2*np.pi*40.3*stec/(C*f),
            'dispersion_ns': 40.3*stec/C*((f-bandwidth_hz/2)**-2-(f+bandwidth_hz/2)**-2)*1e9}


def transfer(stec_tecu, frequency_hz):
    return np.exp(2j*np.pi*40.3*np.asarray(stec_tecu)*1e16/(C*frequency_hz))
