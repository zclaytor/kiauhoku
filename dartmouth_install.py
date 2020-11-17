import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


name = 'dartmouth'
path_to_raw_grids = 'path/to/grid/dartmouth'

filelist = [f for f in os.listdir(path_to_raw_grids) if '.trk' in f]

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

eep_functions = {'prems': my_PreMS, 'rgbump': my_RGBump}
metric_function = my_HRD

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

def all_from_dartmouth(progress=True):
    df_list = []

    if progress:
        file_iter = tqdm(filelist)
    else:
        file_iter = filelist

    for fname in file_iter:
        fpath = os.path.join(path_to_raw_grids, fname)
        df_list.append(from_dartmouth(fpath))

    dfs = pd.concat(df_list).sort_index()
    # Need X_core for EEP computation
    dfs['X_core'] = 1 - dfs['Y_core'] - dfs['Z_core']

    return dfs    

def setup():
    return all_from_dartmouth()

if __name__ == '__main__':
    df = setup()