import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from kiauhoku import stargrid

name = 'yrec'
col_name = 'column_labels.txt'

path_to_raw_grids = '/path/to/raw/grids'
col_path = os.path.join(path_to_raw_grids, col_name)

filelist = [f for f in os.listdir(path_to_raw_grids) if '.track' in f]

# Assign labels used in eep conversion
eep_params = dict(
    name = name,
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


def read_columns(path):
    with open(path, 'r') as f:
        columns = [l.strip() for l in f.readlines()]
    
    return columns

def parse_filename(filename):
    file_str = filename.replace('.track', '')

    mass = float(file_str[:4].replace('_', '.'))

    met_i = file_str.find('fh') + 2
    met_str = file_str[met_i:met_i+4]
    met = float(met_str[1:])/100
    if  met != 0 and met_str[0] == 'm':
        met *= -1

    alpha_i = file_str.find('al') + 2
    alpha_str = file_str[alpha_i:alpha_i+2]
    alpha = float(alpha_str)/10

    return mass, met, alpha


def from_yrec(path, columns=None):
    if columns is None:
        columns = read_columns(col_path)

    fname = path.split('/')[-1]
    initial_mass, initial_met, initial_alpha = parse_filename(fname)

    data = np.loadtxt(path)
    s = np.arange(len(data))
    m = np.ones_like(s) * initial_mass
    z = np.ones_like(s) * initial_met
    a = np.ones_like(s) * initial_alpha

    # Build multi-indexed DataFrame, dropping unwanted columns
    multi_index = pd.MultiIndex.from_tuples(zip(m, z, a, s),
        names=['initial_mass', 'initial_met', 'initial_alpha', 'step'])
    df = pd.DataFrame(data, index=multi_index, columns=columns)
    df = df.drop(columns=[c for c in columns if '#' in c])

    return df

def all_from_yrec(progress=True):
    df_list = []

    if progress:
        file_iter = tqdm(filelist)
    else:
        file_iter = filelist

    for fname in file_iter:
        fpath = os.path.join(path_to_raw_grids, fname)
        df_list.append(from_yrec(fpath))

    dfs = pd.concat(df_list).sort_index()

    return dfs 

def setup():
    return all_from_yrec()

if __name__ == '__main__':
    df = setup()    