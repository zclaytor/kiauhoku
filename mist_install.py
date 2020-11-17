import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


name = 'mist'
path_to_raw_grids = 'path/to/raw/grids'

def from_mist(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    init_names1 = lines[3].strip('#\n').split()
    init_values1 = lines[4].strip('#\n').split()
    init_names2 = lines[6].strip('#\n').split()
    init_values2 = lines[7].strip('#\n').split()

    initial_met = float(init_values1[init_names1.index('[Fe/H]')])
    initial_mass = float(init_values2[init_names2.index('initial_mass')])

    columns = lines[11].strip('#\n').split()
    data = np.genfromtxt(lines[12:])

    s = np.arange(len(data))
    m = np.ones_like(s) * initial_mass
    z = np.ones_like(s) * initial_met

    # Build multi-indexed DataFrame, dropping unwanted columns
    multi_index = pd.MultiIndex.from_tuples(zip(m, z, s),
        names=['initial_mass', 'initial_met', 'eep'])
    df = pd.DataFrame(data, index=multi_index, columns=columns)

    return df

def all_from_mist(progress=True):
    filelist = []
    for folder in os.listdir(path_to_raw_grids):
        path = os.path.join(path_to_raw_grids, folder)
        files = [os.path.join(path, f) for f in os.listdir(path) if '.eep' in f]
        filelist += files

    df_list = []

    if progress:
        file_iter = tqdm(filelist)
    else:
        file_iter = filelist

    df_list = []
    for fname in file_iter:
        try:
            df_list.append(from_mist(fname))
        except:
            print(f'Error reading {fname}---skipping.')
    dfs = pd.concat(df_list).sort_index()

    return dfs    

def setup():
    return all_from_mist()

if __name__ == '__main__':
    df = setup()