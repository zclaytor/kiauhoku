"""
model_grid_utils.py
(C) Zachary R. Claytor
Institute for Astronomy
University of Hawaiʻi
2019 July 1

Python utilities designed to interact with output model grids from the Yale
Rotating stellar Evolution Code (YREC, Demarque et al. 2008). The grid of
models that accompany this software were produced by van Saders & 
Pinsonneault (2013) and updated by Claytor et al. (2019, in prep) using Castelli & 
Kurucz (2004) model atmospheres, which tabulate opacity information for various
metallicities and allow for alpha-enhancement. The rotation code within YREC
assumes a stellar-wind-driven braking law, and specific details on braking
parameters can be found in the file `braking_law_parameters.txt`.

These models span the following mass, metallicity, and alpha-enhancement
(grid parameters can be changed in `config.py`):

              |  min | max | step
    -----------------------------
      M/Msun  |  0.3 | 2.0 | 0.01
      [M/H]   | -1.0 | 0.5 | 0.5
    [alpha/M] |  0.0 | 0.4 | 0.4
    -----------------------------

According to Castelli & Kurucz (2004), [M/H] is log10(Z/Zsun), and 
alpha includes O, Ne, Mg, Si, S, Ar, Ca, and Ti.

The YREC output files have the format `met_###_alpha_###.out`, where ###
specifies the metallicity and alpha-enhancement for the models therein. The
formatting is such that met_-050_alpha_040.out corresponds to the set of tracks
with [M/H] = -0.5 and [alpha/M] = 0.4. Any given `.out` file contains an
evolution track for every mass in the grid. More details on these `.out` files
can be found in the files themselves.
"""


import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from .config import mass_grid, met_grid, alpha_grid
from .config import model_path, column_labels
from .config import eep_path, primary_eep_indices


def pickle_tracks(modelfile, columnfile=model_path+"column_labels.txt"):
    """
    Takes YREC output files for a set of evolution tracks and converts
    them to pandas DataFrames, then saves them to pickles in the same
    directory as the model files.
    Each YREC output file will yield a set of pickle files, one for each
    evolutionary track contained in the model file.

    PARAMETERS
    ----------
    `modelfile`: a string containing the full path to the desired model file

    `columnfile`: a string containing the full path to the file listing the
                  column names to be assigned in the output DataFrame. Each
                  line in this file should contain an single label, and each
                  column in the output file must have a corresponding label.
                  Any columns that the user does not wish to be saved in the 
                  pickled track should have a `#` somewhere on the label line
                  in the columnfile.

    RETURNS nothing.
    """

    with open(columnfile, "r") as cf:
        # Read column labels, but we will use only labels with no '#'.
        column_labels = np.asarray([line.strip() for line in cf.readlines()])
        masked = np.asarray(["#" in label for label in column_labels])

    with open(modelfile, "r") as f:
        header = f.readline()
        # Header format: ' NUMBER OF TRACKS XYZ ...'
        # Each file contains `ntracks` evolutionary tracks, each with
        # `nsteps` steps
        ntracks = int(header[18:21])
        nsteps = np.zeros(ntracks, dtype=int)
        # initial mass and period also specified in preamble
        mass_init = np.zeros(ntracks, dtype=float)
        period_init = np.zeros(ntracks, dtype=float)

        # read preamble. first column is an unnecessary index.
        for i in range(ntracks):
            line = f.readline().split()
            _, nsteps[i], mass_init[i], period_init[i] = line

       
        # read the column label line, but don't use it
        dummy_labels = f.readline()

        # begin reading tracks
        for i in range(ntracks):
            # put together output filename
            mass_str = "_mass_%s.pkl" %_to_string(mass_init[i])
            out_fname = modelfile.replace(".out", mass_str)

            track = np.zeros((nsteps[i], len(column_labels)))
            for j in range(nsteps[i]):
               track[j] = f.readline().split()

            # put track into DataFrame, leaving out unwanted columns, then save.
            df_track = pd.DataFrame(track[:,~masked], columns=column_labels[~masked])
            df_track.to_pickle(out_fname)


def get_full_track(mass, met, alpha, labels=None, 
        read_path=model_path, return_fname=False):
    """
    Obtains the desired stellar evolutionary track from the corresponding
    pickle file

    PARAMETERS
    ----------
    `mass`: (float) the mass of the star on the desired track, 
            in solar mass units

    `met`: (float) the metallicity ([M/H]) of the star on the desired track

    `alpha`: (float) the alpha-enhancement ([alpha/M]) of the star on 
             the desired track

    `labels`: (list of str) the column labels for desired stellar parameters.
              Default is None, which returns all parameters.

    `models_path`: a string containing the path to the directory containing 
                   the model pickle files. Default is "models/".

    `return_fname`: if True, returns the name of the file being read. This
                    is mostly for convenience when converting to Equivalent-
                    Evolutionary- Point- (EEP) based tracks, where the filename
                    is different only by the "eep" prefix. Default is False.

    RETURNS
    -------
    `track`: a pandas DataFrame containing the specified evolutionary track

    `fname`: (optional) the name of the file containing the desired track
    """
    
    # convert input values to strings as they appear in filenames
    mass_str = _to_string(mass)
    met_str = _to_string(met)
    alpha_str = _to_string(alpha)

    fname = "met_%s_alpha_%s_mass_%s.pkl" %(met_str, alpha_str, mass_str)
    track = pd.read_pickle(read_path+fname)
    if labels is not None:
        track = track[labels]

    if return_fname:
        return track, fname
    return track


def _to_string(val):
    """
    Converts a given float (`val`) of mass, metallicity, or alpha 
    enhancement to a string (`my_str`) formatted for the model filename.
    For example, a metallicity [M/H] = -0.5 corresponds to the string 
    "-050", and the mass 1.32 corresponds to the string "132".
    """
    if val < 0:
        my_str = "-"
    else:
        my_str = ""

    my_str += "%03.f" %abs(100*val)
    return my_str
           

def _pickle_series(save_path=model_path):
    """
    Pickles all the models in the grid with specified metallicities
    and alpha enhancements.
    """
    metallicities = [_to_string(met) for met in met_grid]
    alphas = [_to_string(alf) for alf in alpha_grid]

    n_total = len(metallicities)*len(alphas)
    with tqdm(total=n_total) as pbar:
        for met in metallicities:
            for alf in alphas:
                fname = save_path + "met_%s_alpha_%s.out" %(met, alf)
                pickle_tracks(fname)
                pbar.update()


def _pickle_pool(save_path=model_path):
    """
    Pickles all the models in the grid with specified metallicities
    and alpha enhancements.
    """
    metallicities = [_to_string(met) for met in met_grid]
    alphas = [_to_string(alf) for alf in alpha_grid]

    fnames = []
    for met in metallicities:
        for alf in alphas:
            fnames.append(save_path + "met_%s_alpha_%s.out" % (met, alf))

    print("Pickling evolution tracks...")
    with Pool() as pool:
        with tqdm(total=len(fnames)) as pbar:
            for i, _ in enumerate(pool.imap_unordered(pickle_tracks, fnames)):
                pbar.update()


def pickle_all_tracks(use_pool=False):
    """
    Wrapper for functions to pickle evolution tracks.
    Allows user to pickle in series or in parallel using multiprocessing.Pool.
    """
    if use_pool:
        _pickle_pool()
    else:
        _pickle_series()


def get_eep_track(mass, met, alpha, labels="all",
                  re_index=None):
    """
    Given mass, metallicity, and alpha-enhancement, we read and return an
    Equivalent-Evolutionary-Phase- (EEP) based track from file with desired
    column labels.

    User can optionally reindex the EEP-based track. Indices outside
    the current range will be set to NaN.
    """
    met_str = _to_string(met)
    alpha_str = _to_string(alpha)
    mass_str = _to_string(mass)
    fname = "eep_met_%s_alpha_%s_mass_%s.pkl" %(met_str, alpha_str, mass_str)

    # eep_path is defined in eep_config.py
    try:
        if labels == "all":
            eep_track = pd.read_pickle(eep_path+fname)
        else:
            eep_track = pd.read_pickle(eep_path+fname)[labels]
        if re_index is not None:
            eep_track = eep_track.reindex(range(re_index))
        return eep_track
    except FileNotFoundError:
        return np.nan


def _import_model_grid(labels="all"):
    """Gets the grid of all EEP-based evolution tracks; return as nested list.
    """

    re_index = primary_eep_indices[-1]+1
    grid_tracks = [[[get_eep_track(m, z, a, labels, re_index=re_index) 
                     for a in alpha_grid]
                    for z in met_grid]
                   for m in mass_grid]
    return grid_tracks


if __name__ == "__main__":
    pickle_all_tracks(use_pool=True)
