import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from kiauhoku import stargrid


name = 'fastlaunch'

path_to_raw_grids = '/home/zach/Desktop/dev/' + name
filelist = [f for f in os.listdir(path_to_raw_grids) if '.out' in f]

# Assign labels used in eep conversion
eep_params = dict(
    name = name,
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

def _to_string(val):
    '''
    Converts a given float (`val`) of mass, metallicity, or alpha enhancement
    to a string (`my_str`) formatted for the model filename. For example, a
    metallicity [M/H] = -0.5 corresponds to the string '-050', and the mass
    1.32 corresponds to the string '132'.
    '''
    if val < 0:
        my_str = '-'
    else:
        my_str = ''
    my_str += f'{abs(100*val):03.0f}'
    return my_str

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

    return stargrid.from_pandas(df, name)

def all_from_rotevol(progress=True):
    df_list = []

    if progress:
        file_iter = tqdm(filelist)
    else:
        file_iter = filelist

    for fname in file_iter:
        fpath = os.path.join(path_to_raw_grids, fname)
        df_list.append(from_rotevol(fpath))

    dfs = pd.concat(df_list).sort_values(['initial_alpha', 'initial_met', 'initial_mass'])

    return dfs    

def tests():
    assert(_to_string(-0.5) == '-050')
    assert(_to_string(1.32) == '132')
    return all_from_rotevol()

def setup():
    return tests()

if __name__ == '__main__':
    df = tests()
