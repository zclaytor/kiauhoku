"""
eep_functions.py
(C) Zachary R. Claytor
Institute for Astronomy
University of Hawaiʻi
2019 July 1

Python utilities to convert stellar evolution tracks to downsampled tracks
based on Equivalent Evolutionary Phases (EEPs) according to the method of Dotter
(2016). 
"""

from multiprocessing import Pool
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import eep_intervals, primary_eep_indices, eep_path, eep_log_path
from .model_grid_utils import get_full_track, get_eep_track


def eep_interpolate(track, special=False):
    """
    Given a raw evolutionary track, returns a downsampled track based on
    Equivalent Evolutionary Phases (EEPs). The primary EEPs are defined in the
    function `PrimaryEEPs`, and the secondary EEPs are computed based on the
    number of secondary EEPs between each pair of primary EEPs as specified
    in the list `EEP_intervals`. If one of the EEP_intervals is 200, then
    for that pair of primary EEPs, the metric distance between those primary
    EEPs is divided into 200 equally spaced points, and the relevant stellar
    parameters are linearly interpolated at those points.
    """

    i_eep = _primary_eeps(track, special)  # get primary EEP indices in raw track
    num_intervals = len(i_eep) - 1
    # In some cases, the raw models do not hit the ZAMS. In these cases,
    # return -1 for now and make a note.
    if num_intervals == 0:
        return -1

    dist = _Metric_Function(track) # compute metric distance along track

    primary_eep_dist = dist[i_eep]
    secondary_eep_dist = np.zeros(sum(eep_intervals[:num_intervals]) + len(i_eep))

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
    interp_fs = [interp1d(dist, track[col]) for col in track.columns]

    # Interpolate stellar parameters along evolutionary track for
    # desired EEP distances
    eep_track = np.array([f(secondary_eep_dist) for f in interp_fs]).T
    eep_track = pd.DataFrame(eep_track, columns=track.columns)

    return eep_track


def _primary_eeps(track, special=False):
    """
    Given a track, returns a list containing indices of Equivalent
    Evolutionary Phases (EEPs)
    """

    # define a list of functions to iterate over
    if special:
        functions = [_PreMS, _ZAMS, _EAMS, _IAMS, _TAMS, _RGBump_special]
    else:
        functions = [_PreMS, _ZAMS, _EAMS, _IAMS, _TAMS, _RGBump]

    # get indices of EEPs
    i_eep = np.zeros(len(functions)+1, dtype=int)
    for i in range(1,len(i_eep)):
        i_eep[i] = functions[i-1](track, i0=i_eep[i-1])
        if i_eep[i] == -1:
            return i_eep[1:i]
    
    return i_eep[1:]


def _PreMS(track, logTc=5.0, i0=0):
    """
    The pre-main sequence EEP is the point where central temperature rises
    above a certain value (which must be lower than necessary for sustained
    fusion). The default value is log10(T_c) = 5.0, but may be chosen to be
    a different value. An optional argument i0 can be supplied, which is the
    index to start with.

    This relies on the behavior of pandas.Series.argmax() for a Series
    of bools. If no temperature is greater than or equal to logTc, the 
    natural return value is i0. So we don't mistake this failed search,
    we must check the value at i0 to make sure it satisfies our criterion.

    RETURNS
    -------
    `i_PreMS`: (int) the index of the first element in track[i0: "logT(cen)"]
    greater than logTc.    
    """

    logTc_tr = track.loc[i0:, "logT(cen)"]
    i_PreMS = _first_true_index(logTc_tr >= logTc)
    return i_PreMS


def _ZAMS(track, ZAMS_pref=3, Xc_burned=0.001, Hlum_frac_max=0.999, i0=10):
    """
    The Zero-Age Main Sequence EEP has three different implementations in 
    Dotter's code:
      ZAMS1) the point where the core hydrogen mass fraction has been depleted
             by some fraction (0.001 by default: Xc <= Xmax - 0.001)
      ZAMS2) the point *before ZAMS1* where the hydrogen-burning luminosity 
             achieves some fraction of the total luminosity 
             (0.999 by default: Hlum/lum = 0.999)
      ZAMS3) the point *before ZAMS1* with the highest surface gravity

    ZAMS3 is implemented by default.
    """

    Xc_init = track.loc[0, "Xcen"]
    Xc_tr = track.loc[i0:, "Xcen"]
    ZAMS1 = _first_true_index(Xc_tr <= Xc_init-Xc_burned)
    if ZAMS1 == -1:
        return -1

    if ZAMS_pref == 1:
        return ZAMS1

    if ZAMS_pref == 2:
        Hlum_tr = track.loc[i0:ZAMS1, 'H lum (Lsun)']
        lum_tr = track.loc[i0:ZAMS1, 'L/Lsun']
        Hlum_frac = Hlum_tr/lum_tr
        ZAMS2 = _first_true_index(Hlum_frac >= Hlum_frac_max)
        if ZAMS2 == -1:
            return ZAMS1
        return ZAMS2

    logg_tr = track.loc[0:ZAMS1, "logg"]
    ZAMS3 = logg_tr.idxmax()
    return ZAMS3
  

def _IorT_AMS(track, Xmin, i0):
    """
    The Intermediate- and Terminal-Age Main Sequence (IAMS, TAMS) EEPs both use
    the core hydrogen mass fraction dropping below some critical amount.
    This function encapsulates the main part of the code, with the difference
    between IAMS and TAMS being the value of Xmin.
    """
    Xc_tr = track.loc[i0:, 'Xcen']
    i_eep = _first_true_index(Xc_tr <= Xmin)
    return i_eep


def _EAMS(track, Xmin=0.55, i0=12):
    """
    Early-Age Main Sequence. Without this, the low-mass tracks do not
    reach an EEP past the ZAMS before 15 Gyr.
    """
    i_EAMS = _IorT_AMS(track, Xmin, i0)
    return i_EAMS


def _IAMS(track, Xmin=0.3, i0=12):
    """
    Intermediate-Age Main Sequence exists solely to ensure the convective
    hook is sufficiently sampled.
    Defined to be when the core hydrogen mass fraction drops below some
    critical value. Default: Xc <= 0.3
    """
    i_IAMS = _IorT_AMS(track, Xmin, i0)
    return i_IAMS


def _TAMS(track, Xmin=1e-12, i0=14):
    """
    Terminal-Age Main Sequence, defined to be when the core hydrogen mass
    fraction drops below some critical value. Default: Xc <= 1e-12
    """
    i_TAMS = _IorT_AMS(track, Xmin, i0)
    return i_TAMS


def _RGBump(track, i0=None):
    """
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

    Added 2019/05/28: The default grid has two tracks that *just barely* do
    not reach the RGBump. These tracks will use _RGBump_special. In this
    function, I manually set the final point in these tracks as the RGBump
    to extend their EEPs. This will only affect calculations pas the TAMS
    for stars adjacent to these tracks in the grid, and the errors should be
    negligible (but I have not quantified them).
    """
    
    N = len(track)
    lum_tr = track.loc[i0:, "L/Lsun"]
    logT_tr = track.loc[i0:, "Log Teff(K)"]

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


def _RGBump_special(track, i0=None):
    """
    The Red Giant Bump is an interruption in the increase in luminosity on the 
    Red Giant Branch. It occurs when the hydrogen-burning shell reaches the
    composition discontinuity left from the first convective dredge-up.

    Added 2019/05/28: The default grid has two tracks that *just barely* do
    not reach the RGBump. These tracks will use _RGBump_special. In this
    function, I manually set the final point in these tracks as the RGBump
    to extend their EEPs. This will only affect calculations pas the TAMS
    for stars adjacent to these tracks in the grid, and the errors should be
    negligible (but I have not quantified them).
    """
    
    N = len(track)
    return N-1


def _RGBTip(track, i0=None):
    """
    Red Giant Branch Tip
    Dotter describes the tip of the red giant branch (RGBTip) EEP as
    "the point at which stellar luminosity reaches a maximum---or the stellar
    Teff reaches a minimum---after core H burning is complete but before core
    He burning has progressed significantly."

    Note that the YREC models at the time of this writing nominally end at
    the helium flash, so the RGBTip is unadvisable to use as an EEP.
    """

    Ymin = track.loc[i0, "Ycen"] - 1e-2
    Yc_tr = track.loc[i0:, "Ycen"]
    before_He_burned = (Yc_tr > Ymin)
    if not before_He_burned.any():
        return -1

    lum_tr = track.loc[i0:, "L/Lsun"]
    RGBTip1 = (lum_tr[before_He_burned]).idxmax()

    logT_tr = track.loc[i0:, "Log Teff(K)"]
    RGBTip2 = (logT_tr[before_He_burned]).idxmin()

    RGBTip = min(RGBTip1, RGBTip2)
    return RGBTip


def _Metric_Function(track):
    """
    The Metric Function is used to calculate the distance along the evolution
    track. Traditionally, the Euclidean distance along the track on the
    H-R diagram has been used, but any positive-definite function will work.
    """
    return _HRD_distance(track)


def _HRD_distance(track):
    """
    Distance along the H-R diagram, to be used in the Metric Function.
    Returns an array containing the distance from the beginning of the 
    evolution track for each step along the track, in logarithmic effective
    temperature and logarithmic luminosity space.
    """

    # Allow for scaling to make changes in Teff and L comparable
    Tscale = 20
    Lscale = 1

    logTeff = track["Log Teff(K)"]
    logLum = np.log10(track["L/Lsun"])

    N = len(track)
    dist = np.zeros(N)
    for i in range(1, N):
        temp_dist = (((logTeff.iloc[i] - logTeff.iloc[i-1])*Tscale)**2
                       + ((logLum.iloc[i] - logLum.iloc[i-1])*Lscale)**2)
        dist[i] = dist[i-1] + np.sqrt(temp_dist)

    return dist


def _first_true_index(bools):
    """
    Given a pandas Series of bools, returns the index of the first occurrence
    of `True`. **Index-based, NOT location-based**
    I.e., say x = pd.Series({0: False, 2: False, 4: True}), then
    _first_true_index(x) will return index 4, not the positional index 2.

    If no value in `bools` is True, returns -1.
    """
    if not bools.any():
        return -1
    i_true = bools.idxmax()
    return i_true


def convert_track(mass, met, alpha, log_file=None):
    """
    Given mass, metallicity, and alpha in the grid, converts the corresponding
    evolution track into an EEP-based track.
    """
    
    # Define special cases
    special1 = (mass == 0.84) & (met == 0.0) & (alpha == 0.0)
    special2 = (mass == 0.99) & (met == 0.5) & (alpha == 0.4)
    special = special1 | special2

    track, fname = get_full_track(mass, met, alpha, return_fname=True)
    out_fname = "eep_" + fname
    print("Constructing %s..." % out_fname, end="\r")

    try:
        eep_track = eep_interpolate(track, special)
        # If the track doesn't reach the ZAMS, returns -1.
        # In this case, skip and make a note.
        if eep_track is -1:
            if log_file is None:
                print("Track does not reach ZAMS: %s" %fname)
            else:
                print(fname, file=log_file)
            return

        eep_track.to_pickle(eep_path+out_fname)

    except (ValueError, KeyboardInterrupt) as e:
        print("")
        raise e


def _convert_track_pool(fstring):
    """
    Given mass, metallicity, and alpha in the grid, converts the corresponding
    evolution track into an EEP-based track.
    """

    fsplit = fstring.split("_")
    met   = int(fsplit[1])/100
    alpha = int(fsplit[3])/100
    mass  = int(fsplit[5])/100

    # Define special cases
    special1 = (mass == 0.84) & (met == 0.0) & (alpha == 0.0)
    special2 = (mass == 0.99) & (met == 0.5) & (alpha == 0.4)
    special = special1 | special2

    track = get_full_track(mass, met, alpha)
    out_fname = "eep_" + fstring + ".pkl"

    try:
        eep_track = eep_interpolate(track, special)
        # If the track doesn't reach the ZAMS, returns -1.
        # In this case, skip and make a note.
        if eep_track is -1:
            with open(eep_log_path, "a+") as f:
                print(out_fname, file=f)
            return

        eep_track.to_pickle(eep_path+out_fname)

    except (ValueError, KeyboardInterrupt) as e:
        print("")
        raise e


def _convert_series():
    """
    Takes all model grid points, looks for raw track file at each point, 
    and produces EEP-based track for each point. If any models fail to reach
    ZAMS, the corresponding filename is written in a log file.
    """
    from .config import mass_grid, met_grid, alpha_grid

    failed_fname = eep_path + "failed_eep.txt"

    print("Converting tracks to EEP basis...")
    n_total = len(mass_grid) * len(met_grid) * len(alpha_grid)
    with open(failed_fname, "a+") as log_file:
        with tqdm(total=n_total) as pbar:
            for z in met_grid:
                for alpha in alpha_grid:
                    for m in mass_grid:
                        convert_track(m, z, alpha, log_file=log_file)
                        pbar.update()

    print("Tracks successfully converted to EEPs. See %s for failed tracks." % failed_fname)


def _convert_pool():
    """
    Takes all model grid points, looks for raw track file at each point, 
    and produces EEP-based track for each point. If any models fail to reach
    ZAMS, the corresponding filename is written in a log file.
    """
    from .config import mass_grid, met_grid, alpha_grid
    from .model_grid_utils import _to_string
 
    failed_fname = eep_path + "failed_eep.txt"
    string_list = []
    for z in met_grid:
        for alpha in alpha_grid:
            for m in mass_grid:
                string = "met_%s_alpha_%s_mass_%s" % \
                        (_to_string(z), _to_string(alpha), _to_string(m))
                string_list.append(string)

    print("Converting tracks to EEP basis...")
    with Pool() as pool:
        with tqdm(total=len(string_list)) as pbar:
            for i, _ in enumerate(pool.imap_unordered(_convert_track_pool, string_list)):
                pbar.update()

    print("Tracks successfully converted to EEPs. See %s for failed tracks." % failed_fname)


def convert_all_tracks(use_pool=False):
    """
    Wrapper for functions to convert evolution tracks to EEP basis
    Allows user to convert in series or in parallel using multiprocessing.Pool.
    """
    if use_pool:
        _convert_pool()
    else:
        _convert_series()


def HRD(mass=None, met=None, alpha=None, df_track=None, df_eep=None,
         verbose=True):
    """Checking out a particular model on the HRD.
    """

    if df_track is None:
        labels = ["Log Teff(K)", "L/Lsun", "Xcen", "Age(Gyr)"] 
        teff, lum, xcen, age = labels                          

        track = get_full_track(mass, met, alpha, labels)
        eep   = get_eep_track(mass, met, alpha, labels)
    else:
        track = df_track
        eep = df_eep
        teff, lum = "Log Teff(K)", "L/Lsun"

    if verbose:
        i_ZAMS = 201
        print("Number of EEPs in this track: ", len(eep))
        if len(eep) == 202:   final_eep = "ZAMS"
        elif len(eep) == 253: final_eep = "EAMS"
        elif len(eep) == 354: final_eep = "IAMS"
        elif len(eep) == 455: final_eep = "TAMS"
        elif len(eep) == 606: final_eep = "RGBump"
        else:                 final_eep = "ERROR"
        print("    Final EEP reached: ", final_eep)
        print("    Final model Xcen:  ", track.loc[len(track)-1, xcen])  ##
        print("    Final model age:   ", track.loc[len(track)-1, age])   ##
        print("    Final EEP age:     ", eep.loc[len(eep)-1, age])       ##
        print("    ZAMS age:          ", eep.loc[i_ZAMS, age])           ##

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(track[teff], np.log10(track[lum]))
    eep = eep.reindex(range(primary_eep_indices[-1]+1))
    ax.plot(eep.loc[::10, teff], np.log10(eep.loc[::10, lum]),
            "ko", ms=2)
    ax.plot(eep.loc[primary_eep_indices, teff],
            np.log10(eep.loc[primary_eep_indices, lum]),
            "ko", ms=4)

    ax.set_xlabel(r"log($T_\mathrm{eff}$/K)")
    ax.set_ylabel(r"log($L$/$L_\odot$)")
    ax.set_title(r"$M$ = %.02f; [M/H] = %.01f; [$\alpha$/M] = %.01f" %(mass, met, alpha))
    ax.invert_xaxis()
    fig.show()


def _test_failed_eeps():
    """Opens log file `failed_eep.txt` and plots tracks
    """
    with open(eep_path+"failed_eep.txt", "r") as f:
        for line in f:
            track = pd.read_pickle(model_path + line.strip())
            plt.plot(track["Log Teff(K)"], np.log10(track["L/Lsun"]))
    plt.gca().invert_xaxis()
    plt.show()


def _test_models(masses, mets, alphas):
    """
    Plots multiple tracks and their EEP-based tracks, given lists of mass,
    metallicity, and alpha.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(masses)):
        ax = HRD(ax, masses[i], mets[i], alphas[i])

    ax.invert_xaxis()
    plt.show()


if __name__ == "__main__":
    convert_all_tracks(use_pool=True)
