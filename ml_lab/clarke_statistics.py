"""Empirical envelope statistics, downward LCR and complete fade durations."""
import numpy as np


def fading_statistics(time, amplitude, fd):
    t=np.asarray(time,float);a=np.asarray(amplitude,float);power=a*a
    duration=float(t[-1]-t[0]);rms=float(np.sqrt(power.mean()));normalized=a/max(rms,1e-15)
    edges=np.linspace(0,max(3.,float(a.max())),33);counts,_=np.histogram(a,bins=edges,density=True)
    p_edges=np.linspace(0,max(6.,float(power.max())),33);p_counts,_=np.histogram(power,bins=p_edges,density=True)
    grid=np.linspace(0,max(3.,float(a.max())),200)
    thresholds=np.unique(np.r_[np.linspace(.1,2,30),1.]);lcr=[];afd=[];theory_lcr=[];theory_afd=[];complete=[];censored=[]
    for rho in thresholds:
        below=normalized<rho
        starts=np.flatnonzero(~below[:-1]&below[1:])+1
        stops=np.flatnonzero(below[:-1]&~below[1:])+1
        durations=[float(t[stops[stops>start][0]]-t[start]) for start in starts if np.any(stops>start)]
        crossing=len(starts)/duration
        lcr.append(crossing);afd.append(float(np.mean(durations)) if durations else None)
        complete.append(len(durations));censored.append(int(below[0])+int(below[-1]))
        expected=float(np.sqrt(2*np.pi)*fd*rho*np.exp(-rho*rho))
        theory_lcr.append(expected);theory_afd.append(float(-np.expm1(-rho*rho)/expected) if expected>0 else None)
    frequencies=np.linspace(-fd,fd,203)[1:-1] if fd>0 else np.array([])
    psd=1/(np.pi*fd*np.sqrt(1-(frequencies/fd)**2)) if fd>0 else np.array([])
    return {'power':power.tolist(),'rms_amplitude':rms,
            'amplitude_histogram':{'x':((edges[:-1]+edges[1:])/2).tolist(),'density':counts.tolist(),
                'theory_x':grid.tolist(),'theory_density':(2*grid*np.exp(-grid*grid)).tolist()},
            'power_histogram':{'x':((p_edges[:-1]+p_edges[1:])/2).tolist(),'density':p_counts.tolist(),
                'theory_x':np.linspace(0,p_edges[-1],200).tolist(),'theory_density':np.exp(-np.linspace(0,p_edges[-1],200)).tolist()},
            'thresholds_rms':thresholds.tolist(),'lcr_hz':lcr,'afd_s':afd,'complete_fades':complete,'censored_fades':censored,
            'lcr_theory_hz':theory_lcr,'afd_theory_s':theory_afd,
            'theory_psd':{'f':frequencies.tolist(),'db':(10*np.log10(psd)).tolist()},
            'note':'LCR: нисходящие пересечения / длительность. AFD: средняя длительность только завершённых замираний; краевые незавершённые интервалы исключены. Порог нормирован на RMS данной реализации. Теоретические PDF и PSD используют ансамблевую мощность 1.'}
