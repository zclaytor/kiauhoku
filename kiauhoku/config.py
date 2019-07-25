"""
config.py
(C) Zachary R. Claytor
Institute for Astronomy
University of Hawaiʻi
2019 July 1

Configuration file for stellar model grid as implemented in
model_grid_utils.py and eep_functions.py

Here, the user can set paths to import files and destinations. Users working
with their own model grids can set grid and EEP parameters below.

"""

import os
from numpy import arange, cumsum, linspace


# Set path
root = os.path.split(os.path.realpath(__file__))[0] + '/'

# to raw model files
model_path = root + 'grids/'

# Set path to file containing column names.
# NOTE THAT ONLY THE NAMES IN THIS FILE ARE SUPPORTED AT THIS TIME.
columnfile_path = model_path + 'column_labels.txt'

# Set the path to the EEP-based track files
eep_path = model_path

# Creating the EEP tracks will output a log file. Set its path here
eep_log_path = eep_path + 'failed_eep.txt'

# Set the path to the model grid interpolator
interpolator_path = eep_path + 'grid_interpolator.pkl'

# # # # #
# You don't need to change anything below this line

# Set grid parameters here
_mass_min,  _mass_max,  _mass_step  =  0.3, 2.0, 0.01
_met_min,   _met_max,   _met_step   = -1.0, 0.5, 0.5
_alpha_min, _alpha_max, _alpha_step =  0.0, 0.4, 0.4


# `eep_intervals` is a list containing the number of secondary Equivalent
# Evolutionary Phases (EEPs) between each pair of primary EEPs.
# This is intended to be directly manipulated by the user.
eep_intervals = [200, # Between PreMS and ZAMS
                  50, # Between ZAMS and EAMS #150, # ZAMS-IAMS
                 100, # Between EAMS and IAMS
                 100, # IAMS-TAMS
                 150] # TAMS-RGBump


# # # # #
# No one needs to change anything below this line

# Read in column labels
with open(columnfile_path, "r") as f:
    column_labels = [line.strip() for line in f.readlines() if line[0]!="#"]


def grid(gmin, gmax, gstep):
    """Equivalent to np.linspace, but uses step size instead of n_vals
    """
    n_vals = int((gmax - gmin)/gstep + 1)
    my_grid = linspace(gmin, gmax, n_vals)
    return my_grid


# Compute arrays containing grid values
mass_grid = grid(_mass_min, _mass_max, _mass_step)
met_grid = grid(_met_min, _met_max, _met_step)
alpha_grid = grid(_alpha_min, _alpha_max, _alpha_step)

# Derived from EEP_intervals, `primary_eep_indices` is a numpy array
# of the primary EEP indices for each EEP-based track.
primary_eep_indices = arange(len(eep_intervals)+1) 
primary_eep_indices[1:] += cumsum(eep_intervals)

# Unpack primary_eep_indices to name the specific EEPs,
# putting them into the namespace.
i_PreMS, i_ZAMS, i_EAMS, i_IAMS, i_TAMS, i_RGBump = primary_eep_indices 

# Another derived quantity, num_EEP is the total number of EEPs in a track
num_eep = primary_eep_indices[-1] + 1
