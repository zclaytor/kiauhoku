'''
stargrid.py

Contains classes and functions to interact with and interpolate stellar
evolutionary model grids
'''

import os
from importlib import import_module
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import emcee

from .eep import _eep_interpolate
from .interp import DFInterpolator


grids_path = os.path.expanduser('~/') + '.kiauhoku/grids/'

class StarGrid(pd.DataFrame):
    '''
    StarGrid is designed to store and interact with stellar evolution tracks.
    It is little more than a pandas DataFrame with a few extra features.

    Parameters
    ----------
    name (str): the name of the grid, e.g., 'mist'

    eep_params (dict): for EEP-based grids, eep_params contains a mapping from
        your grid's specific column names to the names used by kiauhoku's
        default EEP functions. It also contains 'eep_intervals', the number
        of secondary EEPs between each consecutive pair of primary EEPs.
    '''

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        eep_params = kwargs.pop('eep_params', None)

        # use the __init__ method from DataFrame to ensure
        # that we're inheriting the correct behavior
        super(StarGrid, self).__init__(*args, **kwargs)

        self._metadata = ['name', 'eep_params']
        # Set StarGrid name
        self.name = name
        self.eep_params = eep_params

    # this method makes it so our methods return an instance
    # of StarGrid instead of a regular DataFrame
    @property
    def _constructor(self):
        return StarGrid

    def __setattr__(self, attr, val):
        # have to special case custom attributes because
        # pandas tries to set as columns
        if attr in self._metadata:
            object.__setattr__(self, attr, val)
        else:
            super(StarGrid, self).__setattr__(attr, val)

    def set_name(self, name):
        '''Set the name of the grid.'''
        self.name = name

    def get_track(self, index):
        '''Get a track at a specific index.

        Parameters
        ----------
        index (tuple): corresponds to the input indices of the grid. For
            example, if your grid is indexed by mass, metallicity, and alpha-
            abundance, to get the track for mass=1, metallicity=0.5, alpha=0,

            >>> track = grid.get_track((1, 0.5, 0))

        Returns
        -------
        track (StarGrid): desired evolution track, now indexed only by step.
        '''
        return self.loc[index, :]

    def is_MultiIndex(self):
        '''Checks whether the StarGrid instance is MultiIndexed.'''
        return isinstance(self.index, pd.MultiIndex)

    def to_eep(self,
        eep_params=None,
        eep_functions=None,
        metric_function=None,
        progress=True,
        use_pool=False
    ):
        '''
        Converts the grid of evolution tracks to EEP basis. For details on EEP
        functions, see the documentation for kiauhoku.eep.

        Parameters
        ----------
        eep_params (dict, None): contains a mapping from your grid's specific
            column names to the names used by kiauhoku's default EEP functions.
            It also contains 'eep_intervals', the number of secondary EEPs
            between each consecutive pair of primary EEPs. If none are supplied,
            kiauhoku will attempt to read them from a cache directory.

        eep_functions (dict, None): if the default EEP functions won't do the
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

        progress (bool, True): whether or not to display a tqdm progress bar.

        use_pool (bool, False): (not implemented) whether or not to use
            multiprocessing/pooling.

        Returns
        -------
        eep_frame (StarGrid): grid of EEP-based evolution tracks.
        '''

        if use_pool:
            print('Pooling not yet implemented in <function StarGrid.to_eep>')

        # User can specify eep_params, but if none are specified,
        # searched for cached params.
        if not eep_params:
            eep_params = load_eep_params(self.name)

        # If self is a MultiIndexed DataFrame, split it into individual
        # tracks, convert to EEP basis, and recombine.
        if self.is_MultiIndex():
            idx = self.index.droplevel(-1).drop_duplicates()
            if progress:
                idx_iter = tqdm(idx)
            else:
                idx_iter = idx

            eep_list = []
            idx_list = []

            # Iterate through tracks
            for i in idx_iter:
                eep_track = _eep_interpolate(
                    self.loc[i, :],
                    eep_params,
                    eep_functions,
                    metric_function
                )
                if eep_track is None:
                    continue
                eep_list.append(eep_track)
                idx_list += [(*i, j_eep) for j_eep in eep_track.index]

            # Create MultiIndex for EEP frame
            multiindex = pd.MultiIndex.from_tuples(
                idx_list,
                names=[*idx.names, 'eep']
            )

            eep_frame = pd.concat(eep_list, ignore_index=True)
            eep_frame.index = multiindex

        else:
            eep_frame = _eep_interpolate(
                self, eep_params, eep_functions, metric_function
            )

        eep_frame = from_pandas(eep_frame, name=self.name, eep_params=eep_params)

        return eep_frame

    def get_primary_eeps(self):
        '''Return indices of Primary EEPs in the EEP-based tracks.
        '''
        if 'eep' not in self.index.names:
            raise RuntimeError('Grid is wrong kind. Must be EEP grid.')

        ints = [0] + self.eep_params['intervals']
        eeps = np.arange(len(ints)) + np.cumsum(ints)
        
        return eeps

    def get_eep_track_lengths(self):
        '''
        This is mainly a convenience function to be used in the script
        `eep_track_lengths.py`, but that script is currently configured to work
        only for the rotevol grids.

        Returns
        -------
        lengths: pandas DataFrame containing the number of EEPs in each track
            of the grid.
        '''

        if 'eep' not in self.index.names:
            raise RuntimeError('Grid is wrong kind. Must be EEP grid.')

        idx = self.index.droplevel('eep').drop_duplicates()
        lengths = [len(self.loc[i]) for i in idx]
        lengths = pd.DataFrame(lengths, index=idx)
        return lengths

class StarGridInterpolator(DFInterpolator):
    '''
    Stellar model grid interpolator. Built on the DataFrame Interpolator
    (DFInterpolator) of Tim Morton's isochrones package, the
    StarGridInterpolator is intended to provide easy interaction with stellar
    model grids.

    Attributes
    ----------
    name (str): name of the grid

    columns (list-like): the available columns in the grid.

    max_eep (int): the maximum EEP index out of all the tracks.

    eep_params (dict): the parameters used to calculate the EEPs.
    '''

    def __init__(self, grid):
        super(StarGridInterpolator, self).__init__(grid)

        self.name = grid.name
        self.columns = grid.columns
        self.index = grid.index

        self.max_eep = grid.index.to_frame().eep.max()
        self.eep_params = grid.eep_params

    def get_primary_eeps(self):
        '''Return indices of Primary EEPs in the EEP-based tracks.
        '''
        ints = [0] + self.eep_params['intervals']
        eeps = np.arange(len(ints)) + np.cumsum(ints)
        
        return eeps

    def get_star_eep(self, index):
        '''
        Interpolate a single model from the grid.
        Note that this is the preferred way to sample models from the grid.

        `index` should be a tuple of indices in the same way you would access
        a model from a StarGrid. If your grid is indexed by mass and
        metallicity, and you want the 350th EEP of a 0.987-solar-mass,
        0.2-metallicity star:

        >>> star = grid.get_star_eep((0.987, 0.2, 350))
        '''

        star_values = self(index)
        if len(np.shape(index)) == 1:
            star = pd.Series(star_values, index=self.columns)
        else:
            star = pd.DataFrame(star_values, columns=self.columns)

        return star

    def get_star_age(self, index, age, age_label=None):
        '''
        Interpolate a single model from the grid, accessing by age.
        Note that this method is slower than get_star_eep. get_star_age
        interpolates an entire track from the grid, then runs a 1-D
        interpolator over the track to get the parameters for the desired
        age. get_star_eep is preferred to this method.

        `index` should be a tuple of indices in the same way you would access
        a model from a StarGrid. If your grid is indexed by mass and
        metallicity, and you want a a 4.5-Gyr-old, 0.987-solar-mass,
        0.2-metallicity star:

        >>> star = grid.get_star_age((0.987, 0.2), 4.5)

        Optional Arguments
        ------------------
        age_label (str, None): ideally, you should specify what your grid calls
            age in eep_params in your setup file. If you did, then get_star_age
            can figure out what to call 'age' from the eep_params that are
            stored in the interpolator. If you didn't do this, you can specify
            what your grid calls age using age_label. If grid uses 'Age(Gyr)':

            >>> star = grid.get_star_age(
                    (0.987, 0.2), 4.5, age_label='Age(Gyr)'
                )
        '''

        track = self.get_track(index)
        labels = track.columns
        if age_label is None:
            eep_params = self.eep_params
            if eep_params is None:
                raise ValueError(
                    'No eep_params are stored. Please specify age_label.'
                )
            else:
                age_label = eep_params['age']

        interpf = interp1d(track[age_label], track.values.T)
        star = pd.Series(interpf(age), labels)
        return star

    def get_track(self, index):
        '''
        Interpolate a single track from the model grid.

        `index` should be a tuple of indices in the same way you would access
        a model from a StarGrid. If your grid is indexed by mass and
        metallicity, and you want a track for a 0.987-solar-mass,
        0.2-metallicity star:

        >>> star = grid.get_track((0.987, 0.2))
        '''

        num_eeps = self.max_eep + 1
        ones_arr = np.ones(num_eeps)
        idx = [i*ones_arr for i in index] + [np.arange(num_eeps)]
        star_values = self(idx)
        track = StarGrid(star_values, columns=self.columns,
                         name=self.name, eep_params=self.eep_params)
        return track

    def mcmc_star(self, log_prob_fn, args,
        pos0=None, initial_guess=None, guess_width=None,
        n_walkers=None, n_burnin=0, n_iter=500,
        save_path=None, **kw,
    ):
        '''
        Uses emcee to sample stellar models from the grid.
        For example usage, see mcmc.ipynb in the parent kiauhoku directory.

        Parameters
        ----------
        log_prob_fn (callable): the log-probability function to be passed
            to the emcee EnsembleSampler. Should have call signature
            log_prob_fn(pos, interp, ...), where `pos` is the walker position,
            `interp` is the StarGridInterpolator, and other arguments can be
            supplied as needed using `args`.
            log_prob_fn should return (1), the computed log-probability as a
            float, and (2), the sampled star model from the interpolator. This
            allows `blobs` to be used, and for you to keep track other stellar
            parameters not directly used in the sampling.
            See the docs for emcee for more advanced usage.

        args (tuple): extra arguments to be passed to log_prob_fn.

        pos0 (numpy ndarray, shape n_dim x n_walkers):
            You can optionally directly supply the EnsembleSampler the initial
            walker positions. Alternatively, you can supply a single walker
            position `initial_guess` and take `n_walkers` samples from a
            gaussian distribution with width `guess_width`.

        initial_guess (tuple, optional): initial walker position, to be sampled
            n_walkers times. Should be the same shape as the model grid index.
            Use as an alternative to `pos0`.

        guess_width (tuple, optional): width of initial guess sampling. Should
            be the same shape as the model grid index. Use as an alternative 
            to `pos0`.

        n_walkers (int, optional): number of walkers. If pos0 is specified,
            n_walkers is inferred from the shape of pos0. Otherwise,
            defaults to 12.

        n_burnin (int, optional): number of burn-in steps. Default: 0.

        n_iter (int, optional): number of sample steps. Default: 500.

        save_path (str, optional): You may optionally specify a path to a
            CSV or Parquet file to save the sampler output as a DataFrame.
            Use of Parquet requires that you have pyarrow or another parquet-
            compatible package installed.

        kw: Extra keyword arguments to pass to the EnsembleSampler.

        Returns
        -------
        sampler, the emcee.EnsembleSampler object

        output, a pandas DataFrame comprised of the flattened Markov chains
            from the sampler, plus all the stellar parameters returned from
            each interpolated sample.
        '''

        # If pos0 is not specified, construct it from initial_guess and width
        if pos0 is None:
            if n_walkers is None:
                n_walkers = 12

            pos0 = np.array([
                np.random.normal(guess, width, n_walkers)
                for guess, width in zip(initial_guess, guess_width)
            ]).T

        elif n_walkers is None:
            n_walkers = len(pos0)

        sampler = emcee.EnsembleSampler(
            n_walkers,
            len(initial_guess),
            log_prob_fn,
            args=(self, *args),
            blobs_dtype=[('star', pd.Series)],
            **kw,
        )

        # Run burn-in stage
        if n_burnin > 0:
            pos, prob, state, blobs = sampler.run_mcmc(pos0, n_burnin, progress=True)
            sampler.reset()
        else:
            pos = pos0

        # Run sampling stage
        pos, prob, state, blobs = sampler.run_mcmc(pos, n_iter, progress=True)

        samples = pd.DataFrame(sampler.flatchain, columns=self.index.names)
        blobs = sampler.get_blobs(flat=True)
        blobs = pd.concat(blobs['star'], axis=1).T

        # Concatenate Markov chains with blobs
        output = pd.concat([samples, blobs], axis=1)

        # Save output if desired
        if save_path:
            if 'csv' in save_path:
                output.to_csv(save_path, index=False)
            elif 'pqt' in save_path:
                output.to_parquet(save_path, index=False)
            else:
                print(
                    'save_path extension not recognized, so chains were not saved:\n'
                    f'    {save_path}\n'
                    'Accepted extensions are .csv and .pqt.'
                )

        return sampler, output

    def fit_star(self, star_dict, guess, bounds, *args,
                 loss='meansquarederror', **kw
    ):
        '''
        Fit a star from data using scipy.optimize.minimize.

        PARAMETERS
        ----------
        star_dict: dict containing label-value pairs for the star to be fit

        guess: tuple containing initial guess of input values for star.
            These should be of the same form as the input to
            StarGridInterpolator.get_star_eep.

        bounds: a sequence of (min, max) tuples for each input parameter.

        args: extra arguments to be passed to the loss function.

        kw: extra keyword arguments to be passed to scipy.optimize.minimize.

        RETURNS
        -------
        star: pandas.Series of StarGridInterpolator output for result.

        result: the output of scipy.optimize.minimize.
        '''

        if loss == 'meansquarederror':
            loss_function = self._meansquarederror
        elif loss == 'meanpercenterror':
            loss_function = self._meanpercenterror

        args = (star_dict, *args)
        result = minimize(loss_function, guess, args=args, bounds=bounds, **kw)
        star = self.get_star_eep(result.x)

        return star, result

    def _meansquarederror(self, index, star_dict, scale=False):
        '''Mean Squared Error loss function for `fit_star`.

        Parameters
        ----------
        index (tuple): index to be fit.

        star_dict (dict): dictionary of values for loss function computation.

        scale (list-like, optional): Optionally scale the squared errors before
            taking the mean. This could be useful if, for example, luminosity is
            in solar units (~1) and age is in years (~10^9 years).

        Returns
        -------
        mean squared error as a float.
        '''

        star = self.get_star_eep(index)
        sq_err = np.array([(star[l] - star_dict[l])**2 for l in star_dict])

        if scale:
            sq_err /= np.array(scale)**2

        return np.average(sq_err)

    def _meanpercenterror(self, index, star_dict):
        '''Mean Percent Error loss function for `fit_star`.

        Parameters
        ----------
        index (tuple): index to be fit.

        star_dict (dict): dictionary of values for loss function computation.

        Returns
        -------
        mean percent error as a float.
        '''

        star = self.get_star_eep(index)
        mpe = np.average(
            [np.abs(star[l] - star_dict[l])/star_dict[l] for l in star_dict]
        )
        return mpe

    def _chisq(self, index, star_dict, err_dict, 
        err='average', return_star=False
    ):
        '''Convenience routine to compute the chi-squared of a fit.
        '''
        star = self.get_star_eep(index)

        chisq = 0
        for l in star_dict:
            if isinstance(err_dict[l], (tuple, list, np.array)):
                if err == 'average':
                    uncert = np.average(err_dict[l])
                elif err == 'min':
                    uncert = min(err_dict[l])
                elif err == 'max':
                    uncert = max(err_dict[l])
            else:
                uncert = err_dict[l]

            chisq += ((star[l] - star_dict[l]) / uncert)**2

        if return_star:
            return chisq, star
        return chisq

    def to_pickle(self, path=None):
        '''Saves the StarGridInterpolator to a pickle file.
        '''
        if path is None:
            path = os.path.join(grids_path, self.name, 'interpolator.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)

def load_interpolator(name=None, path=None):
    '''
    Load StarGridInterpolator from pickle file. If the interpolator has been
    cached during the install, simply specifying the name will be enough to
    load it.
    '''

    if name and path:
        raise ValueError('Please specify only `name` or `path`.')
    elif name:
        path = os.path.join(grids_path, name, 'interpolator.pkl')
    elif path:
        pass
    else:
        raise ValueError('Specify `name` or `path`.')
    with open(path, 'rb') as f:
        interp = pickle.load(f)
    return interp

def from_pandas(df, *args, **kwargs):
    '''Convert pandas DataFrame to StarGrid object.
    '''
    return StarGrid(df, *args, **kwargs)

def read_pickle(*args, **kwargs):
    '''Read StarGrid from pickle file.
    '''
    name = kwargs.pop('name', None)
    df = pd.read_pickle(*args, **kwargs)
    return from_pandas(df, name=name)

def read_csv(*args, **kwargs):
    '''Read StarGrid from csv.
    '''
    name = kwargs.pop('name', None)
    df = pd.read_csv(*args, **kwargs)
    return from_pandas(df, name=name)

def read_parquet(*args, **kwargs):
    '''
    Read StarGrid from parquet. Requires installation of pyarrow or
    similar package to support parquet.
    '''
    name = kwargs.pop('name', None)
    eep_params = kwargs.pop('eep_params', None)
    df = pd.read_parquet(*args, **kwargs)
    return from_pandas(df, name=name, eep_params=eep_params)

def install_grid(script, kind='raw'):
    '''
    Installs grid from a user-created setup file. For examples, see
    *_install.py scripts in the kiauhoku parent directory.

    Installation file must have a variable `name` set to the name of the grid,
    as well as a function `setup` that returns the set of stellar model grids
    as a MultiIndexed pandas DataFrame.

    Usage for `rotevol_install.py`:
        >>> import kiauhoku as kh
        >>> kh.install_grid('rotevol_install')
    '''

    # For now, MIST only works if the grids are already in EEP basis.
    if 'mist' in script and kind == 'raw':
        raise NotImplementedError(
            'For now, MIST input grids must already be in EEP basis.\n'
            'Please specify kind="eep".'
        )

    module = import_module(script)
    print(f'Installing grid "{module.name}" from {script}')

    # Create cache directories
    path = os.path.join(grids_path, module.name)
    if not os.path.exists(path):
        os.makedirs(path)

    if kind == 'raw':
        eep_params = module.eep_params
        # Cache eep parameters
        with open(os.path.join(path, 'eep_params.pkl'), 'wb') as f:
            pickle.dump(eep_params, f)

        print('Reading and combining grid files')
        grids = module.setup()
        grids = from_pandas(grids, name=module.name)

        # Save full grid to file
        full_save_path = os.path.join(path, 'full_grid.pqt')
        print(f'Saving to {full_save_path}')
        grids.to_parquet(full_save_path)

        print(f'Converting to eep-based tracks')
        try:
            eep_functions = module.eep_functions
        except AttributeError:
            eep_functions = None
        try:
            metric_function = module.metric_function
        except AttributeError:
            metric_function = None

        eeps = grids.to_eep(eep_params, eep_functions, metric_function)

    elif kind == 'eep':
        eeps = module.setup()
        eeps = from_pandas(eeps, name=module.name)

    # Save EEP grid to file
    eep_save_path = os.path.join(path, 'eep_grid.pqt')
    print(f'Saving to {eep_save_path}')
    eeps.to_parquet(eep_save_path)

    # Create and save interpolator to file
    interp = StarGridInterpolator(eeps)
    interp_save_path = os.path.join(path, 'interpolator.pkl')
    print(f'Saving interpolator to {interp_save_path}')
    interp.to_pickle(path=interp_save_path)

    print(f'Model grid "{module.name}" installed.')

def load_full_grid(name):
    '''Load raw model grid from file.
    '''
    return load_grid(name, kind='full')

def load_eep_grid(name):
    '''Load EEP-based model grid from file.
    '''
    return load_grid(name, kind='eep')

def load_grid(name, kind='full'):
    '''Load model grid from file.
    '''
    file_path = os.path.join(grids_path, name, f'{kind}_grid.pqt')

    if kind == 'eep':
        eep_params = load_eep_params(name)
    else:
        eep_params = None

    if os.path.exists(file_path):
        return read_parquet(file_path, name=name, eep_params=eep_params)
    raise FileNotFoundError(f'{file_path}: No such file exists.')

def load_eep_params(name):
    '''
    Assuming EEP params were specified in the setup script and cached,
    this will load them from the cache by specifying the grid name.
    '''
    params_path = os.path.join(grids_path, name, 'eep_params.pkl')
    with open(params_path, 'rb') as f:
        eep_params = pickle.load(f)

    return eep_params