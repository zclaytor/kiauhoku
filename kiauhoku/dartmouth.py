import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


# Assign labels used in eep conversion
eep_params = dict(
    age = 'Age (yrs)',
    hydrogen_lum = 'L_H',
    lum = 'Log L',
    logg = 'Log g',
    log_teff = 'Log T',
    core_hydrogen_frac = 'X_core', # must be added 
    core_helium_frac = 'Y_core',
    teff_scale = 20, # used in metric function
    lum_scale = 1, # used in metric function
    # `intervals` is a list containing the number of secondary Equivalent
    # Evolutionary Phases (EEPs) between each pair of primary EEPs.
    intervals = [200, # Between PreMS and ZAMS
                  50, # Between ZAMS and EAMS 
                 100, # Between EAMS and IAMS
                 100, # IAMS-TAMS
                 150], # TAMS-RGBump
)

def my_PreMS(track, eep_params, i0=None):
    '''
    Dartmouth models do not have central temperature, which is necessary for
    the default PreMS calculation. For now, let the first point be the PreMS.
    '''
    return 0

def my_TAMS(track, eep_params, i0, Xmin=1e-5):
    '''
    By default, the TAMS is defined as the first point in the track where Xcen
    drops below 10^-12. But not all the DSEP tracks hit this value. To ensure
    the TAMS is placed correctly, here I'm using Xcen = 10^-5 as the critical
    value.
    '''
    core_hydrogen_frac = eep_params['core_hydrogen_frac']
    Xc_tr = track.loc[i0:, core_hydrogen_frac]
    below_crit = Xc_tr <= Xmin
    if not below_crit.any():
        return -1
    return below_crit.idxmax()

def my_RGBump(track, eep_params, i0=None):
    '''
    Modified from eep.get_RGBump to make luminosity logarithmic
    '''

    lum = eep_params['lum']
    log_teff = eep_params['log_teff']
    N = len(track)

    lum_tr = track.loc[i0:, lum]
    logT_tr = track.loc[i0:, log_teff]

    lum_greater = (lum_tr > 1)
    if not lum_greater.any():
        return -1
    RGBump = lum_greater.idxmax() + 1

    while logT_tr[RGBump] < logT_tr[RGBump-1] and RGBump < N-1:
        RGBump += 1

    # Two cases: 1) We didn't reach an extremum, in which case RGBump gets
    # set as the final index of the track. In this case, return -1.
    # 2) We found the extremum, in which case RGBump gets set
    # as the index corresponding to the extremum.
    if RGBump >= N-1:
        return -1
    return RGBump-1

def my_HRD(track, eep_params):
    '''
    Adapted from eep._HRD_distance to fix lum logarithm
    '''

    # Allow for scaling to make changes in Teff and L comparable
    Tscale = eep_params['teff_scale']
    Lscale = eep_params['lum_scale']

    log_teff = eep_params['log_teff']
    lum = eep_params['lum']

    logTeff = track[log_teff]
    logLum = track[lum]

    N = len(track)
    dist = np.zeros(N)
    for i in range(1, N):
        temp_dist = (((logTeff.iloc[i] - logTeff.iloc[i-1])*Tscale)**2
                    + ((logLum.iloc[i] - logLum.iloc[i-1])*Lscale)**2)
        dist[i] = dist[i-1] + np.sqrt(temp_dist)

    return dist

def from_dartmouth(path):
    fname = path.split('/')[-1]
    file_str = fname.replace('.trk', '')

    mass = int(file_str[1:4])/100

    met_str = file_str[7:10]
    met = int(met_str[1:])/10
    if met_str[0] == 'm':
        met *= -1

    alpha_str = file_str[13:]
    alpha = int(alpha_str[1:])/10
    if alpha_str[0] == 'm':
        alpha *= -1
   
    with open(path, 'r') as f:
        header = f.readline()
        col_line = f.readline()
        data_lines = f.readlines()
    
    columns = re.split(r'\s{2,}', col_line.strip('# \n'))
    data = np.genfromtxt(data_lines)
    
    # Build multi-indexed DataFrame, dropping unwanted columns
    multi_index = pd.MultiIndex.from_tuples(
        [(mass, met, step) for step in range(len(data))],
        names=['initial_mass', 'initial_met', 'step'])
    df = pd.DataFrame(data, index=multi_index, columns=columns)

    return df

def all_from_dartmouth(raw_grids_path, progress=True):
    df_list = []
    filelist = [f for f in os.listdir(raw_grids_path) if '.trk' in f]

    if progress:
        file_iter = tqdm(filelist)
    else:
        file_iter = filelist

    for fname in file_iter:
        fpath = os.path.join(raw_grids_path, fname)
        df_list.append(from_dartmouth(fpath))

    dfs = pd.concat(df_list).sort_index()
    # Need X_core for EEP computation
    dfs['X_core'] = 1 - dfs['Y_core'] - dfs['Z_core']

    return dfs    

def install(
    raw_grids_path,
    name=None,
    eep_params=eep_params,
    eep_functions={'prems': my_PreMS, 'tams': my_TAMS, 'rgbump': my_RGBump},
    metric_function=my_HRD,
    ):
    '''
    The main method to install grids that are output of the `rotevol` rotational
    evolution tracer code.

    Parameters
    ----------
    raw_grids_path (str): the path to the folder containing the raw model grids.

    name (str, optional): the name of the grid you're installing. By default,
        the basename of the `raw_grids_path` will be used.

    eep_params (dict, optional): contains a mapping from your grid's specific
        column names to the names used by kiauhoku's default EEP functions.
        It also contains 'eep_intervals', the number of secondary EEPs
        between each consecutive pair of primary EEPs. By default, the params
        defined at the top of this script will be used, but users may specify
        their own.

    eep_functions (dict, optional): if the default EEP functions won't do the
        job, you can specify your own and supply them in a dictionary.
        EEP functions must have the call signature
        function(track, eep_params), where `track` is a single track.
        If none are supplied, the default functions will be used.

    metric_function (callable, None): the metric function is how the EEP
        interpolator spaces the secondary EEPs. By default, the path
        length along the evolution track on the H-R diagram (luminosity vs.
        Teff) is used, but you can specify your own if desired.
        metric_function must have the call signature
        function(track, eep_params), where `track` is a single track.
        If no function is supplied, defaults to dartmouth.my_HRD.

    Returns None
    '''
    from .stargrid import from_pandas
    from .stargrid import grids_path as install_path

    if name is None:
        name = os.path.basename(raw_grids_path)

    # Create cache directories
    path = os.path.join(install_path, name)
    if not os.path.exists(path):
        os.makedirs(path)

    # Cache eep parameters
    with open(os.path.join(path, 'eep_params.pkl'), 'wb') as f:
        pickle.dump(eep_params, f)

    print('Reading and combining grid files')
    grids = all_from_dartmouth(raw_grids_path)
    grids = from_pandas(grids, name=name)

    # Save full grid to file
    full_save_path = os.path.join(path, 'full_grid.pqt')
    print(f'Saving to {full_save_path}')
    grids.to_parquet(full_save_path)

    print(f'Converting to eep-based tracks')
    eeps = grids.to_eep(eep_params, eep_functions, metric_function)

    # Save EEP grid to file
    eep_save_path = os.path.join(path, 'eep_grid.pqt')
    print(f'Saving to {eep_save_path}')
    eeps.to_parquet(eep_save_path)

    # Create and save interpolator to file
    interp = eeps.to_interpolator()
    interp_save_path = os.path.join(path, 'interpolator.pkl')
    print(f'Saving interpolator to {interp_save_path}')
    interp.to_pickle(path=interp_save_path)

    print(f'Model grid "{name}" installed.')