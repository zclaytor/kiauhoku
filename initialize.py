"""
setup.py
(C) Zachary R. Claytor
Institute for Astronomy
University of Hawaiʻi
2019 July 1

Setup script to turn stellar evolution tracks into Equivalent-Evolutionary
Phase (EEP)-based tracks. Takes the full evolution tracks in .txt files and
outputs EEP-based tracks in .pkl files. Finally creates model grid interpolator
and dumps it into a pickle file for easy use.
"""

from .model_grid_utils import pickle_all_tracks
from .eep_functions import convert_all_tracks
from .Grid import pickle_interpolator

def initialize():
    use_pool = True
    pickle_all_tracks(use_pool)
    convert_all_tracks(use_pool)
    pickle_interpolator()
