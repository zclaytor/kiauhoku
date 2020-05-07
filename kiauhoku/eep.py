'''
eep.py

Utilities to downsample stellar evolution tracks to Equivalent Evolutionary
Phase (EEP) basis, according to the method of Dotter (2016). 

The default EEP functions are contained in eep.default_eep_functions. They are
    default_eep_functions = {
        'prems': get_PreMS,
        'zams': get_ZAMS,
        'eams': get_EAMS,
        'iams': get_IAMS,
        'tams': get_TAMS,
        'rgbump': get_RGBump,
    }

You can define and supply your own EEP functions in a dictionary. 
EEP functions must have the call signature
    function(track, eep_params)
where `track` is a single track.
'''

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def _eep_interpolate(track, eep_params, eep_functions, metric_function=None):
    '''
    Given a raw evolutionary track, returns a downsampled track based on
    Equivalent Evolutionary Phases (EEPs). The primary EEPs are defined below,
    and the default ones to use in the computation are listed in
    eep.default_eep_functions. The secondary EEPs are computed based on the
    number of secondary EEPs between each pair of primary EEPs as specified
    in `eep_params.intervals`; these should be defined in the grid
    installation file. If one of the EEP_intervals is 200, then for that pair
    of primary EEPs, the metric distance between those primary EEPs is divided
    into 200 equally spaced points, and the relevant stellar parameters are
    linearly interpolated at those points.

    Parameters
    ----------
    track (StarGrid): single-index StarGrid to be interpolated.

    eep_params (dict): dictionary of column names to use in EEP computation, 
        as well as secondary EEP intervals.

    eep_functions (dict): dictionary of callables to use for EEP computation.

    metric_function (callable, optional): function to compute EEP intervals.
        If none is specified, eep._HRD_distance will be used.

    Returns
    -------
    eep_track, a pandas DataFrame containing the EEP-based track.
    '''

    i_eep = _locate_primary_eeps(track, eep_params, eep_functions)
    num_intervals = len(i_eep) - 1
    # In some cases, the raw models do not hit the ZAMS. In these cases,
    # return None.
    if num_intervals == 0:
        return

    if metric_function is None:
        metric_function = _HRD_distance

    # compute metric distance along track
    dist = metric_function(track, eep_params)

    primary_eep_dist = dist[i_eep]
    eep_intervals = eep_params['intervals']
    secondary_eep_dist = np.zeros(
        sum(eep_intervals[:num_intervals]) + len(i_eep)
    )

    # Determine appropriate distance to each secondary EEP
    j0 = 0
    for i in range(num_intervals):
        my_dist = primary_eep_dist[i+1] - primary_eep_dist[i]
        delta = my_dist/(eep_intervals[i] + 1)
        new_dist = np.array([primary_eep_dist[i] + delta*j \
                for j in range(eep_intervals[i]+1)])
        
        secondary_eep_dist[j0:j0+len(new_dist)] = new_dist
        j0 += len(new_dist)

    secondary_eep_dist[-1] = primary_eep_dist[-1]

    # Create list of interpolator functions
    interp = interp1d(dist, track.T)

    # Interpolate stellar parameters along evolutionary track for
    # desired EEP distances
    eep_track = pd.DataFrame(interp(secondary_eep_dist).T, columns=track.columns)
    eep_track.index.name = 'eep'

    return eep_track

def _locate_primary_eeps(track, eep_params, eep_functions=None):
    '''
    Given a track, returns a list containing indices of Equivalent
    Evolutionary Phases (EEPs)
    '''

    # define a list of functions to iterate over
    eep_f = default_eep_functions
    if eep_functions is None:
        eep_functions = default_eep_functions
    else:
        eep_f.update(eep_functions)        

    # get indices of EEPs
    i_eep = []
    i_start = 0
    for f in eep_f.values():
        i_phase = f(track, eep_params, i0=i_start)
        i_eep.append(i_phase)
        i_start = i_phase
        if i_start == -1:
            return np.array(i_eep[:-1])
    
    return np.array(i_eep)
    
def get_PreMS(track, eep_params, i0=0, logTc_crit=5.0):
    '''
    The pre-main sequence EEP is the point where central temperature rises
    above a certain value (which must be lower than necessary for sustained
    fusion). The default value is log10(T_c) = 5.0, but may be chosen to be
    different. An optional argument i0 can be supplied, which is the
    index to start with.

    This relies on the behavior of pandas.Series.idxmax() for a Series
    of bools. If no temperature is greater than or equal to logTc, the 
    natural return value is i0. So we don't mistake this failed search,
    we must check the value at i0 to make sure it satisfies our criterion.

    Returns
    -------
    `i_PreMS`: (int) index of the first element in track[i0: "logT(cen)"]
    greater than logTc.    
    '''
    log_central_temp = eep_params['log_central_temp']

    logTc_tr = track.loc[i0:, log_central_temp]
    i_PreMS = _first_true_index(logTc_tr >= logTc_crit)
    return i_PreMS

def get_ZAMS(track, eep_params, i0=10, ZAMS_pref=3, Xc_burned=0.001, 
    Hlum_frac_max=0.999):
    '''
    The Zero-Age Main Sequence EEP has three different implementations in 
    Dotter's code:
    ZAMS1) the point where the core hydrogen mass fraction has been depleted
            by some fraction (0.001 by default: Xc <= Xmax - 0.001)
    ZAMS2) the point *before ZAMS1* where the hydrogen-burning luminosity 
            achieves some fraction of the total luminosity 
            (0.999 by default: Hlum/lum = 0.999)
    ZAMS3) the point *before ZAMS1* with the highest surface gravity

    ZAMS3 is implemented by default.
    '''
    core_hydrogen_frac = eep_params['core_hydrogen_frac']
    Xc_init = track.at[0, core_hydrogen_frac]
    Xc_tr = track.loc[i0:, core_hydrogen_frac]
    ZAMS1 = _first_true_index(Xc_tr <= Xc_init-Xc_burned)

    if ZAMS1 == -1:
        return -1

    if ZAMS_pref == 1:
        return ZAMS1

    if ZAMS_pref == 2:
        hydrogen_lum = eep_params['hydrogen_lum']
        lum = eep_params['lum']

        Hlum_tr = track.loc[i0:ZAMS1, hydrogen_lum]
        lum_tr = track.loc[i0:ZAMS1, lum]
        Hlum_frac = Hlum_tr/lum_tr
        ZAMS2 = _first_true_index(Hlum_frac >= Hlum_frac_max)
        if ZAMS2 == -1:
            return ZAMS1
        return ZAMS2

    # or ZAMS_pref = 3
    logg = eep_params['logg']
    logg_tr = track.loc[0:ZAMS1, logg]
    ZAMS3 = logg_tr.idxmax()
    return ZAMS3

def get_IorT_AMS(track, eep_params, i0, Xmin):
    '''
    The Intermediate- and Terminal-Age Main Sequence (IAMS, TAMS) EEPs both use
    the core hydrogen mass fraction dropping below some critical amount.
    This function encapsulates the main part of the code, with the difference
    between IAMS and TAMS being the value of Xmin.
    '''
    core_hydrogen_frac = eep_params['core_hydrogen_frac']
    Xc_tr = track.loc[i0:, core_hydrogen_frac]
    i_eep = _first_true_index(Xc_tr <= Xmin)
    return i_eep 

def get_EAMS(track, eep_params, i0=12, Xmin=0.55):
    '''
    Early-Age Main Sequence. Without this, the low-mass rotevol tracks do not
    reach an EEP past the ZAMS before 15 Gyr.
    '''
    i_EAMS = get_IorT_AMS(track, eep_params, i0, Xmin)
    return i_EAMS

def get_IAMS(track, eep_params, i0=12, Xmin=0.3):
    '''
    Intermediate-Age Main Sequence exists solely to ensure the convective
    hook is sufficiently sampled.
    Defined to be when the core hydrogen mass fraction drops below some
    critical value. Default: Xc <= 0.3
    '''
    i_IAMS = get_IorT_AMS(track, eep_params, i0, Xmin)
    return i_IAMS

def get_TAMS(track, eep_params, i0=14, Xmin=1e-12):
    '''
    Terminal-Age Main Sequence, defined to be when the core hydrogen mass
    fraction drops below some critical value. Default: Xc <= 1e-12
    '''
    i_TAMS = get_IorT_AMS(track, eep_params, i0, Xmin)
    return i_TAMS

def get_RGBump(track, eep_params, i0=None):
    '''
    The Red Giant Bump is an interruption in the increase in luminosity on the 
    Red Giant Branch. It occurs when the hydrogen-burning shell reaches the
    composition discontinuity left from the first convective dredge-up.

    Dotter skips the Red Giant Bump and proceeds to the Tip of the Red Giant
    Branch, but since the YREC models, at the time of this writing, terminate
    at the helium flash, I choose to use the Red Giant Bump as my final EEP.

    I identify the RGBump as the first local minimum in Teff after the TAMS.
    To avoid weird end-of-track artifacts, if the minimum is within 1 step
    from the end of the raw track, the track is treated as if it doesn't reach
    the RGBump.

    Added 2018/07/22: Some tracks have weird, jumpy behavior before the RGBump
    which gets mistakenly identified as the RGBump. To avoid this, I force the
    RGBump to be the first local minimum in Teff after the TAMS *and* with
    a luminosity above 10 Lsun.
    '''

    lum = eep_params['lum']
    log_teff = eep_params['log_teff']
    N = len(track)

    lum_tr = track.loc[i0:, lum]
    logT_tr = track.loc[i0:, log_teff]

    RGBump = _first_true_index(lum_tr > 10) + 1
    if RGBump == 0:
        return -1

    while logT_tr[RGBump] < logT_tr[RGBump-1] and RGBump < N-1:
        RGBump += 1

    # Two cases: 1) We didn't reach an extremum, in which case RGBump gets
    # set as the final index of the track. In this case, return -1.
    # 2) We found the extremum, in which case RGBump gets set
    # as the index corresponding to the extremum.
    if RGBump >= N-1:
        return -1
    return RGBump-1

def get_RGBTip(track, eep_params, i0=None):
    '''
    Red Giant Branch Tip
    Dotter describes the tip of the red giant branch (RGBTip) EEP as
    "the point at which stellar luminosity reaches a maximum---or the stellar
    Teff reaches a minimum---after core H burning is complete but before core
    He burning has progressed significantly."

    Note that the YREC models at the time of this writing nominally end at
    the helium flash, so the RGBTip is not recommended to use as an EEP.
    '''

    core_helium_frac = eep_params['core_helium_frac']
    lum = eep_params['lum']
    log_teff = eep_params['log_teff']

    Ymin = track.at[i0, core_helium_frac] - 1e-2
    Yc_tr = track.loc[i0:, core_helium_frac]
    before_He_burned = (Yc_tr > Ymin)
    if not before_He_burned.any():
        return -1

    lum_tr = track.loc[i0:, lum]
    RGBTip1 = (lum_tr[before_He_burned]).idxmax()

    logT_tr = track.loc[i0:, log_teff]
    RGBTip2 = (logT_tr[before_He_burned]).idxmin()

    RGBTip = min(RGBTip1, RGBTip2)
    return RGBTip

def get_ZACHeB(track, eep_params, i0=None):
    '''
    Zero-Age Core Helium Burning
    Denotes the onset of sustained core He burning. The point is identified 
    as the core temperature minimum that occurs after the onset of He burning
    (RGBTip) while Ycen > Ycen,RGBTip - 0.03.
    '''

    core_helium_frac = eep_params['core_helium_frac']
    log_central_temp = eep_params['log_central_temp']
        
    Ymin = track.at[i0, core_helium_frac] - 0.03
    Yc_tr = track.loc[i0:, core_helium_frac]
    before_He_burned = (Yc_tr > Ymin)
    
    if not before_He_burned.any():
        return -1

    logTc_tr = track.loc[i0:, log_central_temp]
    ZACHeB = (logTc_tr[before_He_burned]).idxmin()

    return ZACHeB

def get_TACHeB(track, eep_params, i0=None, Yc_crit=1e-4):
    '''
    Terminal-Age Core Helium Burning
    Corresponds to Ycen = 1e-4.
    '''

    core_helium_frac = eep_params['core_helium_frac']
    Yc_tr = track.loc[i0:, core_helium_frac]

    TACHeB = _first_true_index(Yc_tr < Yc_crit)
    return TACHeB

def get_TPAGB(track, eep_params, i0=None):
    '''
    Thermally-Pulsing Asymptotic Giant Branch
    Not yet implemented.
    '''

    raise NotImplementedError(
        'Function "get_TPAGB" not yet implemented.'
    )

def get_PostAGB(track, eep_params, i0=None):
    '''
    Post AGB
    Not yet implemented.
    '''

    raise NotImplementedError(
        'Function "get_PostTPAGB" not yet implemented.'
    )

def get_WDCS(track, eep_params, i0=None):
    '''
    White Dwarf Cooling Sequence
    Not yet implemented.
    '''

    raise NotImplementedError(
        'Function "get_WDCS" not yet implemented.'
    )


def _HRD_distance(track, eep_params):
    '''
    The Metric Function is used to calculate the distance along the evolution
    track. Traditionally, the Euclidean distance along the track on the
    H-R diagram has been used, but any positive-definite function will work.
    
    Distance along the H-R diagram, to be used in the Metric Function.
    Returns an array containing the distance from the beginning of the 
    evolution track for each step along the track, in logarithmic effective
    temperature and logarithmic luminosity space.
    '''

    # Allow for scaling to make changes in Teff and L comparable
    Tscale = eep_params['teff_scale']
    Lscale = eep_params['lum_scale']

    log_teff = eep_params['log_teff']
    lum = eep_params['lum']

    logTeff = track[log_teff]
    logLum = np.log10(track[lum])

    N = len(track)
    dist = np.zeros(N)
    for i in range(1, N):
        temp_dist = (((logTeff.iloc[i] - logTeff.iloc[i-1])*Tscale)**2
                    + ((logLum.iloc[i] - logLum.iloc[i-1])*Lscale)**2)
        dist[i] = dist[i-1] + np.sqrt(temp_dist)

    return dist

def _first_true_index(bools):
    '''
    Given a pandas Series of bools, returns the index of the first occurrence
    of `True`. **Index-based, NOT location-based**
    I.e., say x = pd.Series({0: False, 2: False, 4: True}), then
    _first_true_index(x) will return index 4, not the positional index 2.

    If no value in `bools` is True, returns -1.
    '''
    if not bools.any():
        return -1
    return bools.idxmax()

default_eep_functions = {
    'prems': get_PreMS,
    'zams': get_ZAMS,
    'eams': get_EAMS,
    'iams': get_IAMS,
    'tams': get_TAMS,
    'rgbump': get_RGBump,
}