import os
from importlib import import_module
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import emcee

try:
    from isochrones.interp import DFInterpolator
except ImportError:
    print (
        'Use of kiauhoku requires installation of isochrones:\n'
        'https://isochrones.readthedocs.io/en/latest/install.html\n'
    )
    raise

from .eep import _eep_interpolate


grids_path = os.path.expanduser('~/') + '.kiauhoku/grids/'

class StarGrid(pd.DataFrame):
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
        self.name = name

    def get_track(self, index):
        return self.loc[index, :]
   
    def is_MultiIndex(self):
        return isinstance(self.index, pd.MultiIndex)

    def to_eep(self, eep_params=None, eep_functions=None, metric_function=None,
        progress=True, use_pool=False):

        if use_pool:
            print('Pooling not yet implemented in <function StarGrid.to_eep>')
        
        if not eep_params:
            eep_params = load_eep_params(self.name)

        if self.is_MultiIndex():
            idx = self.index.droplevel('step').drop_duplicates()
            if progress:
                idx_iter = tqdm(idx)
            else:
                idx_iter = idx

            eep_list = []
            idx_list = []
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

            multiindex = pd.MultiIndex.from_tuples(idx_list,
                names=idx.names+['eep'])

            eep_frame = pd.concat(eep_list, ignore_index=True)
            eep_frame.index = multiindex

        else:
            eep_frame = _eep_interpolate(
                self, eep_params, eep_functions, metric_function
            )

        eep_frame = from_pandas(eep_frame, name=self.name, eep_params=eep_params)

        return eep_frame

    def get_eep_track_lengths(self):
        if 'eep' not in self.index.names:
            raise KeyError('Grid is wrong kind. Must be EEP grid.')

        idx = self.index.droplevel('eep').drop_duplicates()
        lengths = [len(self.loc[i]) for i in idx]
        lengths = pd.DataFrame(lengths, index=idx)
        return lengths

class StarGridInterpolator(DFInterpolator):
    def __init__(self, grid):
        super(StarGridInterpolator, self).__init__(grid)

        self.name = grid.name
        self.columns = grid.columns

        self.max_eep = grid.index.to_frame().eep.max()
        self.eep_params = grid.eep_params

    def get_star_eep(self, index):
        star_values = self(index)
        return pd.Series(star_values, index=self.columns)

    def get_star_age(self, index, age, age_label=None):
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

    def mcmc_star(
        self, log_prob_fn, args,
        initial_guess, guess_width,
        n_walkers=12, n_burnin=0, n_iter=500,
        save_path=None
    ):

        pos0 = np.array([
            np.random.normal(initial_guess[l], guess_width[l], n_walkers)
            for l in initial_guess
        ]).T

        sampler = emcee.EnsembleSampler(n_walkers, len(initial_guess),
            log_prob_fn=log_prob_fn, 
            args=(self,) + args,
            vectorize=False,
            blobs_dtype=[('star', pd.Series)]
        )

        if n_burnin > 0:
            pos, prob, state, blobs = sampler.run_mcmc(pos0, n_burnin, progress=True)
            sampler.reset()
        else:
            pos = pos0

        pos, prob, state, blobs = sampler.run_mcmc(pos, n_iter, progress=True)

        samples = pd.DataFrame(sampler.flatchain, columns=initial_guess.keys())
        blobs = sampler.get_blobs(flat=True)
        blobs = pd.concat(blobs['star'], axis=1).T

        output = pd.concat([samples, blobs], axis=1)

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

    def fit_star(self, star_dict,
                 guess, 
                 bounds,
                 *args,
                 loss='meansquarederror',
                 **kw
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

    def _meansquarederror(self, x, star_dict, scale=False):
        star = self.get_star_eep(x)
        sq_err = np.array([(star[l] - star_dict[l])**2 for l in star_dict])
        
        if scale:
            sq_err /= np.array(scale)

        return np.average(sq_err)

    def _meanpercenterror(self, x, star_dict):
        star = self.get_star_eep(x)
        mpe = np.average(
                [np.abs(star[l] - star_dict[l])/star_dict[l] for l in star_dict]
        )
        return mpe

    def _chisq(self, x, star_dict, err_dict, err='average', return_star=False):
        star = self.get_star_eep(x)

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

    def get_track(self, index):
        num_eeps = self.max_eep + 1
        ones_arr = np.ones(num_eeps)
        idx = [i*ones_arr for i in index] + [np.arange(num_eeps)]
        star_values = self(idx)
        return StarGrid(star_values, columns=self.columns, name=self.name, eep_params=self.eep_params)

    def to_pickle(self, path=None):
        if path is None:
            path = os.path.join(grids_path, self.name, 'interpolator.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)

def load_interpolator(name=None, path=None):
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
    return StarGrid(df, *args, **kwargs)

def read_pickle(*args, **kwargs):
    name = kwargs.pop('name', None)
    df = pd.read_pickle(*args, **kwargs)
    return from_pandas(df, name=name)

def read_csv(*args, **kwargs):
    name = kwargs.pop('name', None)
    df = pd.read_csv(*args, **kwargs)
    return from_pandas(df, name=name)

def read_parquet(*args, **kwargs):
    name = kwargs.pop('name', None)
    eep_params = kwargs.pop('eep_params', None)
    df = pd.read_parquet(*args, **kwargs)
    return from_pandas(df, name=name, eep_params=eep_params)

def install_grid(script, kind='raw'):
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
        except:
            metric_function = None

        eeps = grids.to_eep(eep_params, eep_functions, metric_function)

    elif kind == 'eep':
        eeps = module.setup()
        eeps = from_pandas(eeps, name=module.name)

    eep_save_path = os.path.join(path, 'eep_grid.pqt')
    print(f'Saving to {eep_save_path}')
    eeps.to_parquet(eep_save_path)

    interp = StarGridInterpolator(eeps)
    interp_save_path = os.path.join(path, 'interpolator.pkl')
    print(f'Saving interpolator to {interp_save_path}')
    interp.to_pickle(path=interp_save_path)

    print(f'Model grid "{module.name}" installed.')

def load_full_grid(name):
    return load_grid(name, kind='full')

def load_eep_grid(name):
    eep_params = load_eep_params(name)
    return load_grid(name, eep_params, kind='eep')

def load_grid(name, eep_params=None, kind='full'):
    file_path = os.path.join(grids_path, name, f'{kind}_grid.pqt')
    if os.path.exists(file_path):
        return read_parquet(file_path, name=name, eep_params=eep_params)
    raise FileNotFoundError(f'{file_path}: No such file exists.')

def load_eep_params(name):
    params_path = os.path.join(grids_path, name, 'eep_params.pkl')
    with open(params_path, 'rb') as f: 
        eep_params = pickle.load(f)

    return eep_params