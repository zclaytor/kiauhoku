import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


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

def all_from_mist(raw_grids_path, progress=True):
    filelist = []
    for folder in os.listdir(raw_grids_path):
        path = os.path.join(raw_grids_path, folder)
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

def install(
    raw_grids_path,
    name=None,
    ):
    '''
    The main method to install grids that are output of the `rotevol` rotational
    evolution tracer code.

    Parameters
    ----------
    raw_grids_path (str): the path to the folder containing the raw model grids.

    name (str, optional): the name of the grid you're installing. By default,
        the basename of the `raw_grids_path` will be used.

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

    eeps = all_from_mist(raw_grids_path)
    eeps = from_pandas(eeps, name=name)

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