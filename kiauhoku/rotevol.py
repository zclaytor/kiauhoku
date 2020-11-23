import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


# Assign labels used in eep conversion
eep_params = dict(
    age = 'Age(Gyr)',
    log_central_temp = 'logT(cen)',
    core_hydrogen_frac = 'Xcen',
    hydrogen_lum = 'H lum (Lsun)',
    lum = 'L/Lsun',
    logg = 'logg',
    log_teff = 'Log Teff(K)',
    core_helium_frac = 'Ycen',
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


def from_rotevol(path):
    fname = path.split('/')[-1]
    val_str = fname.replace('.out', '')
    _, met_str, _, alpha_str = val_str.split('_')
    met = float(met_str)/100
    alpha = float(alpha_str)/100
        
    with open(path, 'r') as f:
        header = f.readline()
        # Header format: ` NUMBER OF TRACKS NNN ...`
        # Each file contains `ntracks` evolutionary tracks, each with `nsteps` steps
        
        ntracks = int(header[18:21])
        nsteps = np.zeros(ntracks, dtype=int)
        
        # Initial mass and period also specified in preamble
        mass_init = np.zeros(ntracks, dtype=float)
        period_init = np.zeros(ntracks, dtype=float)
        
        # Read preamble. First column is an unnecessart index.
        for i in range(ntracks):
            line = f.readline().split()
            _, nsteps[i], mass_init[i], period_init[i] = line
            
        # Read the column label line
        columns = re.split(r'\s{2,}', f.readline().strip())
        
        # Begin reading tracks
        df_tracks = []
        for i in range(ntracks):
            # Construct track array
            track = np.zeros((nsteps[i], len(columns)))
            for j in range(nsteps[i]):
                track[j] = f.readline().split()

            # Convert track to DataFrame and save.
            df_track = pd.DataFrame(track, columns=columns)
            df_tracks.append(df_track)

    df_tracks = pd.concat(df_tracks)
    j_track = df_tracks['J'].values - 1
    k_step = df_tracks['K'].values.astype('int') - 1
    m = np.round(0.3 + j_track/100, 2)
    z = np.round(np.ones_like(m) * met, 1)
    a = np.round(np.ones_like(m) * alpha, 1)
    s = k_step

    # Build multi-indexed DataFrame, dropping unwanted columns
    multi_index = pd.MultiIndex.from_tuples(zip(m, z, a, s),
        names=['initial_mass', 'initial_met', 'initial_alpha', 'step'])
    df = pd.DataFrame(df_tracks.values, index=multi_index, columns=columns)
    df = df.drop(columns=['J', 'K', 'dummyGamma', 'dummyCcore'])
    df = df.loc[:, ~df.columns.duplicated()]

    return df

def all_from_rotevol(raw_grids_path, progress=True):
    df_list = []
    filelist = [f for f in os.listdir(raw_grids_path) if '.out' in f]

    if progress:
        file_iter = tqdm(filelist)
    else:
        file_iter = filelist

    for fname in file_iter:
        fpath = os.path.join(raw_grids_path, fname)
        df_list.append(from_rotevol(fpath))

    dfs = pd.concat(df_list).sort_index()

    return dfs    

def install(
    raw_grids_path,
    name=None,
    eep_params=eep_params,
    eep_functions=None,
    metric_function=None,
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
        If no function is supplied, defaults to kiauhoku.eep._HRD_distance.

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
    grids = all_from_rotevol(raw_grids_path)
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