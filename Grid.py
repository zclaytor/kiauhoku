"""
Grid.py
(C) Zachary R. Claytor
Institute for Astronomy
University of Hawaiʻi
2019 July 1

Kīauhōkū
From Hawaiian:
1. vt. To sense the span of a star's existence (i.e., its age).
2. n. The speed of a star (in this case, its rotational speed).

Grid interpolator for rotating stellar models, designed for gyrochronology.

Module containing the Grid class, which is designed to
interpolate stars from a model grid.

Usage:
    >>> labels = ["Age(Gyr)", "Log Teff(K)", "L/Lsun", ...]
    >>> # Getting some star with mass m, metalicity z, alpha a, age t
    >>> grid = load_interpolator(labels)
    >>> star = grid.get_star(m, z, a, age=t)

"""

import numpy as np
import pickle
from scipy.interpolate import RegularGridInterpolator, interp1d
import pandas as pd
from emcee import EnsembleSampler

from .config import mass_grid, met_grid, alpha_grid, column_labels
from .config import num_eep, interpolator_path
from .model_grid_utils import _import_model_grid


class Grid(object):
    """
    A list of scipy.interpolate.RegularGridInterpolator objects, designed
    to interpolate a star of desired mass, metallicity, alpha-enhancement, 
    and age from a grid of models.

    PARAMETERS
    ----------
    `labels` (list of str): The column labels for the desired stellar parameters
        to interpolate. Column labels must match the labels as set in the model
        grid. I.e., they must match the labels in models/column_labels.txt.
    """

    def __init__(self, labels=None, functions=None):
        if labels is None or labels == "all":
            self.labels = column_labels
        else:
            if isinstance(labels, list):
                if len(labels) == 0:
                    self.labels = column_labels
                else:
                    self.labels = labels 

        if functions is None:
            self.functions = _interp_initialize(self.labels)
        else:
            self.functions = functions


    def __str__(self):
        return "Grid interpolator with labels\n" + str(self.labels)


    def __repr__(self):
        return self.__str__()


    def drop(self, labels):
        """Drops labels from the grid interpolator.

        PARAMETERS
        ----------
        `labels` (list): list of labels to be dropped.

        RETURNS
        -------
        Grid object with desired labels kept.

        """
        new_labels = [l for l in self.labels if l not in labels]
        new_functions = [f for f,l in zip(self.functions, self.labels) if l not in labels]

        return Grid(labels=new_labels, functions=new_functions) 


    def keep(self, labels):
        """Keeps specified labels, but drops the rest from the grid interpolator.

        PARAMETERS
        ----------
        `labels` (list): list of labels to be kept.

        RETURNS
        -------
        Grid object with desired labels kept.

        """

        new_labels = [l for l in self.labels if l in labels]
        new_functions = [f for f,l in zip(self.functions, self.labels) if l in labels]

        return Grid(labels=new_labels, functions=new_functions)

    
    def to_pickle(self, fname=interpolator_path):
        """Dumps grid interpolator into pickle file located at `fname`.
        """
        with open(fname, "wb+") as f:
            pickle.dump(self, f)


    def get_eep(self, mass, met, alpha, i_eep, _as_array=False):
        """
        Interpolate a single step in the evolution track. This is the
        simplest interpolation, since the interpolation scheme is built
        on Equivalent Evolutionary Phases (EEPs). It requires a single
        call to the interpolating functions and no further steps.

        PARAMETERS
        ----------
        `mass` (float): desired stellar mass in solar units
        `met` (float): desired metallicity ([M/H])
        `alpha` (float): desired alpha-element enhancement ([alpha/M])
        `i_eep` (float): desired EEP step, basically the evolutionary state

        RETURNS
        -------
        `interps`: a pandas Series containing a star interpolated from the grid

        """

        vals = np.array((mass, met, alpha, i_eep))
        interps = np.array([f(vals).item() for f in self.functions])
        if _as_array:
            return interps
        return pd.Series(interps, index=self.labels)


    def get_track_piece(self, mass, met, alpha, i_start, i_end, _as_array=False):
        """
        Interpolate a sequence of EEPs between (and including)
        `i_start` and `i_end`

        PARAMETERS
        ----------
        `mass` (float): desired stellar mass in solar units
        `met` (float): desired metallicity ([M/H])
        `alpha` (float): desired alpha-element enhancement ([alpha/M])
        `i_start` (int): desired EEP for the beginning of the track
        `i_end` (int): desired EEP for the end of the track

        RETURNS
        -------
        `interps`: a pandas DataFrame containing an evolution track segment 
            interpolated from the grid

        """

        idx = list(range(i_start, i_end+1))
        vals = np.array([[mass, met, alpha, i] for i in idx])
        interps = np.array([f(vals) for f in self.functions]).T
        if _as_array:
            return interps
        return pd.DataFrame(interps, columns=self.labels)


    def get_track(self, mass, met, alpha, _as_array=False):
        """Interpolate an entire evolution track.
        
        PARAMETERS
        ----------
        `mass` (float): desired stellar mass in solar units
        `met` (float): desired metallicity ([M/H])
        `alpha` (float): desired alpha-element enhancement ([alpha/M])

        RETURNS
        -------
        `interps`: a pandas DataFrame containing an evolution track 
            interpolated from the grid

        """

        interps = self.get_track_piece(mass, met, alpha, 0, num_eep-1, _as_array)
        return interps


    def get_star(self, mass, met, alpha, i_eep=None, age=None,
                 label=None, value=None,
                 i_start=0, i_end=num_eep-1,
                 asarray=False):
        """
        The most general interpolator. If `i_eep` is specified, a call to 
        `get_eep` is made. If `age` or `label` is specified an entire track 
        is obtained, and then one further interpolation step is made to get 
        the parameters for the desired age (or `value`) from the track.

        PARAMETERS
        ----------
        `mass` (float): desired stellar mass in solar units
        `met` (float): desired metallicity ([M/H])
        `alpha` (float): desired alpha-element enhancement ([alpha/M])

        While the following parameters are individually optional, one
        or the other MUST be specified.
        `i_eep` (float, optional): desired EEP step
        `age` (float, optional): deisred stellar age
        `label` (string, optional): label of parameter to use as age proxy

        If `label` is specified, then `value` must also be specified.
        `value` (float, optional): desired age proxy value to be interpolated.

        If `label` is specified, `i_start` and `i_end` may also be specified.
        This way, a piece of the track may be used for interpolation instead of the
        entire track. This is useful in cases where the desired parameter is not
        monotonic, e.g., rotation period.

        `asarray` (bool): if True, returns output as numpy array instead of
            pandas Series. Default is False.

        RETURNS
        -------
        `my_star`: a pandas Series containing parameters for a star
            interpolated from the grid

        """
        if i_eep is not None:
            my_star = self.get_eep(mass, met, alpha, i_eep, _as_array=True)

        elif age is not None:
            track = self.get_track_piece(mass, met, alpha, 
                    i_start=i_start, i_end=i_end, _as_array=True)
            i_label = self.labels.index("Age(Gyr)")
            interps = interp1d(track[:, i_label], track.T)

            my_star = interps(age)

        elif label is not None:
            if value is None:
                raise ValueError("If `label` is specified, `value` must also be specified!")

            track = self.get_track_piece(mass, met, alpha,
                                         i_start=i_start, i_end=i_end,
                                         _as_array=True)
            i_label = self.labels.index(label)
            interps = interp1d(track[:, i_label], track.T)

            my_star = interps(value)

        else:
            raise ValueError("One of `i_eep`, `age` or `label` must be specified.")

        if asarray:
            return my_star
        return pd.Series(my_star, index=self.labels)


    def mcmc(self, n_walkers, n_iter, n_burnin, lnprob, args, pos0, chain_labels,
             pool=None, progress=True, out_file=None):
        """
        PARAMETERS
        ----------
        `n_walkers` (int): the number of walkers to use
        `n_iter` (int): the number of sample iterations to perform post burn-in
        `n_burnin` (int): the number of burn-in steps to perform
        `lnprob` (func): function returning the log-posterior probability
        `args` (tuple): arguments to be passed to `lnprob`
        `pos0` (list-like): list of initial walker positions
        `chain_labels` (list of str): list of column labels for the sample chains
        `out_file` (str, optional): the user has the option to save the sample
            chains and blobs to a csv or pickle file. This is the path to the
            output filename.

        RETURNS
        -------
        `output`: a pandas DataFrame containing all the sample chains and blobs

        """
       
        n_dim = len(chain_labels)
        sampler = EnsembleSampler(n_walkers, n_dim, lnprob, args=args,
                                        pool=pool,
                                        blobs_dtype=[("star", pd.Series)])

        # Burn-in phase
        if n_burnin != 0:
            print("Burn-in phase...", end="\r")
            pos, prob, state, blobs = sampler.run_mcmc(pos0, n_burnin)
            sampler.reset()
        else:
            pos = pos0

        # Sampling phase
        pos, prob, state, blobs = sampler.run_mcmc(pos, n_iter, 
                                                   progress=progress)

        samples = pd.DataFrame(sampler.flatchain, columns=chain_labels)
        blobs = sampler.get_blobs(flat=True)
        blobs = pd.concat(blobs["star"], axis=1).T

        output = pd.concat([samples, blobs], axis=1)
        if out_file is not None:
            if "csv" in out_file:
                output.to_csv(out_file, index=False)
            else:
                output.to_pickle(out_file)

        return sampler, output


def from_pickle(fname=interpolator_path):
    """Loads grid inperpolator from pickle located at `fname`.
    """
    with open(fname, "rb") as f:
        grid = pickle.load(f)

    return grid


def load_interpolator(labels=None):
    """
    Loads grid interpolator from default location (specified in eep_config.py),
    keeping only the labels specified by the user. If no labels are specified,
    everything is returned.
    """
    if labels is None or labels == "all":
        return from_pickle()
    return from_pickle().keep(labels)


def _interp_initialize(labels):
    """
    Loads the grid of EEP-based stellar evolution models, then feeds the
    relevant parameters into a list of RegularGridInterpolators.
    Each RegularGridInterpolator has a value of a stellar parameter at each
    point with a given mass, metallicity, alpha-enhancement, and secondary EEP.

    Returns `interp_fs`, a list of RegularGridInterpolators such that

    >>> interp_fs[0](m, z, a, i)
    
    gives the value of whatever stellar parameter's interpolator is first in
    the list for that combination of mass (m), metallicity (z), alpha (a),
    and EEP (i).
    """
    tracks = _import_model_grid(labels)
    vals = [np.array([[[_values(k, lab, num_eep) for k in j] 
                       for j in i] 
                      for i in tracks]) 
            for lab in labels]
    ieep = np.arange(num_eep)

    interp_fs = [RegularGridInterpolator((mass_grid, met_grid, alpha_grid, ieep), v) 
                 for v in vals]
    return interp_fs


def _values(obj, label, n):
    """
    If `obj` is not a pandas DataFrame, returns a numpy array of NaNs.
    If `obj` is a DataFrame, returns the desired column as a numpy array.
    This is designed to ensure trying to interpolate between two tracks
    with different lengths returns NaN when one end of the interpolation
    is undefined.
    """
    if isinstance(obj, pd.DataFrame):
        return obj[label].values
    return np.full(n, np.nan)


def pickle_interpolator():
    """Dumps the interpolator with all labels into a pickle file.
    """
    print("Building and pickling interpolator...")
    grid = Grid(labels="all")
    grid.to_pickle()


if __name__ == "__main__":
    iickle_interpolator()
