import os
from importlib import import_module
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import emcee

from isochrones.interp import DFInterpolator

grids_path = os.path.expanduser('~/') + '.kiauhoku/grids/'

class StarGrid(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)

        # use the __init__ method from DataFrame to ensure
        # that we're inheriting the correct behavior
        super(StarGrid, self).__init__(*args, **kwargs)

        self._metadata = ['name', 'eep_params']
        # Set StarGrid name        
        self.name = name
        self.eep_params = None

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

    def _check_bounds(self, mass, met, alpha):
        mass_min, mass_max = self.get_mass_lim()
        if not (mass_min <= mass <= mass_max):
            raise ValueError(f'Mass {mass} out of range {self.mass_lim}.')
        met_min, met_max = self.get_met_lim()
        if not (met_min <= met <= met_max):
            raise ValueError(f'Metallicity {met} out of range {self.met_lim}.')
        alpha_min, alpha_max = self.get_alpha_lim()
        if not (alpha_min <= alpha <= alpha_max):
            raise ValueError(f'Alpha {alpha} out of range {self.alpha_lim}.')
        return True

    def _get_values_helper(self, column):
        if not self.is_MultiIndex():
            raise ValueError('Grid is not MultiIndex.')
        values = self.index.get_level_values(column).drop_duplicates().values
        return values

    def get_mass_values(self):
        values = self._get_values_helper('initial_mass')
        return values

    def get_met_values(self):
        values = self._get_values_helper('initial_met')
        return values

    def get_alpha_values(self):
        values = self._get_values_helper('initial_alpha') 
        return values

    def get_mass_min(self):
        return self.get_mass_values().min()
        
    def get_mass_max(self):
        return self.get_mass_values().max()

    def get_met_min(self):
        return self.get_met_values().min()

    def get_met_max(self):
        return self.get_met_values().max()

    def get_alpha_min(self):
        return self.get_alpha_values().min()

    def get_alpha_max(self):
        return self.get_alpha_values().max()

    def get_mass_lim(self):
        values = self.get_mass_values()
        return (values.min(), values.max())

    def get_met_lim(self):
        values = self.get_met_values()
        return (values.min(), values.max())

    def get_alpha_lim(self):
        values = self.get_alpha_values()
        return (values.min(), values.max())

    def get_track(self, mass, met, alpha):
        if self._check_bounds(mass, met, alpha):
            return self.loc[mass, met, alpha, :]
   
    def is_MultiIndex(self):
        return isinstance(self.index, pd.MultiIndex)

    def to_eep(self, progress=True, use_pool=False, **eep_params):
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
            for m, z, a in idx_iter:
                eep_track = self.loc[m, z, a, :]._eep_interpolate(**eep_params)
                if eep_track is None:
                    continue
                eep_list.append(eep_track)
                idx_list += [(m, z, a, i) for i in eep_track.index]

            multiindex = pd.MultiIndex.from_tuples(idx_list,
                names=['initial_mass', 'initial_met', 'initial_alpha', 'eep'])

            eep_frame = pd.concat(eep_list, ignore_index=True)
            eep_frame.index = multiindex

        else:
            eep_frame = self._eep_interpolate(**eep_params)

        eep_frame.name = self.name
        eep_frame.eep_params = eep_params

        return eep_frame

    def _eep_interpolate(self, **eep_params):
        '''
        Given a raw evolutionary track, returns a downsampled track based on
        Equivalent Evolutionary Phases (EEPs). The primary EEPs are defined in the
        function `PrimaryEEPs`, and the secondary EEPs are computed based on the
        number of secondary EEPs between each pair of primary EEPs as specified
        in the list `EEP_intervals`. If one of the EEP_intervals is 200, then
        for that pair of primary EEPs, the metric distance between those primary
        EEPs is divided into 200 equally spaced points, and the relevant stellar
        parameters are linearly interpolated at those points.
        '''
  
        if self.eep_params is None:
            self.eep_params = load_eep_params(self.name)

        i_eep = self._locate_primary_eeps()
        num_intervals = len(i_eep) - 1
        # In some cases, the raw models do not hit the ZAMS. In these cases,
        # return None.
        if num_intervals == 0:
            return

        dist = self._Metric_Function() # compute metric distance along track

        primary_eep_dist = dist[i_eep]
        eep_intervals = self.eep_params['intervals']
        secondary_eep_dist = np.zeros(sum(eep_intervals[:num_intervals]) + len(i_eep))

        # Determine appropriate distance to each secondary EEP
        j0 = 0
        for i in range(num_intervals):
            my_dist = primary_eep_dist[i+1] - primary_eep_dist[i]
            delta = my_dist/(eep_intervals[i] + 1)
            new_dist = np.array([primary_eep_dist[i] + delta*j \
                    for j in range(eep_intervals[i]+1)])
            
            secondary_eep_dist[j0:j0+len(new_dist)] = new_dist
            j0 += len(new_dist)

        secondary_eep_dist[-1] = primary_eep_dist[-1]

        # Create list of interpolator functions
        interp_fs = [interp1d(dist, self[col]) for col in self.columns]

        # Interpolate stellar parameters along evolutionary track for
        # desired EEP distances
        eep_track = np.array([f(secondary_eep_dist) for f in interp_fs]).T
        eep_track = pd.DataFrame(eep_track, columns=self.columns)
        eep_track.index.name = 'eep'

        return from_pandas(eep_track)


    def _locate_primary_eeps(self):
        '''
        Given a track, returns a list containing indices of Equivalent
        Evolutionary Phases (EEPs)
        '''
        if not self.eep_params:
            eep_params = load_eep_params(self.name)

        # define a list of functions to iterate over
        functions = [self.get_PreMS, 
                     self.get_ZAMS, 
                     self.get_EAMS, 
                     self.get_IAMS, 
                     self.get_TAMS, 
                     self.get_RGBump]

        # get indices of EEPs
        i_eep = np.zeros(len(functions)+1, dtype=int)
        for i in range(1,len(i_eep)):
            i_eep[i] = functions[i-1](i0=i_eep[i-1])
            if i_eep[i] == -1:
                return i_eep[1:i]
        
        return i_eep[1:]
        
    def get_PreMS(self, i0=0, logTc_crit=5.0):
        '''
        The pre-main sequence EEP is the point where central temperature rises
        above a certain value (which must be lower than necessary for sustained
        fusion). The default value is log10(T_c) = 5.0, but may be chosen to be
        different. An optional argument i0 can be supplied, which is the
        index to start with.

        This relies on the behavior of pandas.Series.argmax() for a Series
        of bools. If no temperature is greater than or equal to logTc, the 
        natural return value is i0. So we don't mistake this failed search,
        we must check the value at i0 to make sure it satisfies our criterion.

        RETURNS
        -------
        `i_PreMS`: (int) index of the first element in track[i0: "logT(cen)"]
        greater than logTc.    
        '''
        log_central_temp = self.eep_params['log_central_temp']
        logTc_tr = self.loc[i0:, log_central_temp]
        i_PreMS = _first_true_index(logTc_tr >= logTc_crit)
        return i_PreMS

    def get_ZAMS(self, i0=10, ZAMS_pref=3, Xc_burned=0.001, 
        Hlum_frac_max=0.999):
        '''
        The Zero-Age Main Sequence EEP has three different implementations in 
        Dotter's code:
        ZAMS1) the point where the core hydrogen mass fraction has been depleted
                by some fraction (0.001 by default: Xc <= Xmax - 0.001)
        ZAMS2) the point *before ZAMS1* where the hydrogen-burning luminosity 
                achieves some fraction of the total luminosity 
                (0.999 by default: Hlum/lum = 0.999)
        ZAMS3) the point *before ZAMS1* with the highest surface gravity

        ZAMS3 is implemented by default.
        '''
        core_hydrogen_frac = self.eep_params['core_hydrogen_frac']
        hydrogen_lum = self.eep_params['hydrogen_lum']
        lum = self.eep_params['lum']
        logg = self.eep_params['logg']

        Xc_init = self.loc[0, core_hydrogen_frac]
        Xc_tr = self.loc[i0:, core_hydrogen_frac]
        ZAMS1 = _first_true_index(Xc_tr <= Xc_init-Xc_burned)
        if ZAMS1 == -1:
            return -1

        if ZAMS_pref == 1:
            return ZAMS1

        if ZAMS_pref == 2:
            Hlum_tr = self.loc[i0:ZAMS1, hydrogen_lum]
            lum_tr = self.loc[i0:ZAMS1, lum]
            Hlum_frac = Hlum_tr/lum_tr
            ZAMS2 = _first_true_index(Hlum_frac >= Hlum_frac_max)
            if ZAMS2 == -1:
                return ZAMS1
            return ZAMS2

        logg_tr = self.loc[0:ZAMS1, logg]
        ZAMS3 = logg_tr.idxmax()
        return ZAMS3

    def get_IorT_AMS(self, i0, Xmin):
        '''
        The Intermediate- and Terminal-Age Main Sequence (IAMS, TAMS) EEPs both use
        the core hydrogen mass fraction dropping below some critical amount.
        This function encapsulates the main part of the code, with the difference
        between IAMS and TAMS being the value of Xmin.
        '''
        core_hydrogen_frac = self.eep_params['core_hydrogen_frac']
        Xc_tr = self.loc[i0:, core_hydrogen_frac]
        i_eep = _first_true_index(Xc_tr <= Xmin)
        return i_eep 

    def get_EAMS(self, i0=12, Xmin=0.55):
        '''
        Early-Age Main Sequence. Without this, the low-mass tracks do not
        reach an EEP past the ZAMS before 15 Gyr.
        '''
        i_EAMS = self.get_IorT_AMS(i0, Xmin)
        return i_EAMS

    def get_IAMS(self, i0=12, Xmin=0.3):
        '''
        Intermediate-Age Main Sequence exists solely to ensure the convective
        hook is sufficiently sampled.
        Defined to be when the core hydrogen mass fraction drops below some
        critical value. Default: Xc <= 0.3
        '''
        i_IAMS = self.get_IorT_AMS(i0, Xmin)
        return i_IAMS

    def get_TAMS(self, i0=14, Xmin=1e-12):
        '''
        Terminal-Age Main Sequence, defined to be when the core hydrogen mass
        fraction drops below some critical value. Default: Xc <= 1e-12
        '''
        i_TAMS = self.get_IorT_AMS(i0, Xmin)
        return i_TAMS

    def get_RGBump(self, i0=None):
        '''
        The Red Giant Bump is an interruption in the increase in luminosity on the 
        Red Giant Branch. It occurs when the hydrogen-burning shell reaches the
        composition discontinuity left from the first convective dredge-up.

        Dotter skips the Red Giant Bump and proceeds to the Tip of the Red Giant
        Branch, but since the YREC models, at the time of this writing, terminate
        at the helium flash, I choose to use the Red Giant Bump as my final EEP.

        I identify the RGBump as the first local minimum in Teff after the TAMS.
        To avoid weird end-of-track artifacts, if the minimum is within 1 step
        from the end of the raw track, the track is treated as if it doesn't reach
        the RGBump.

        Added 2018/07/22: Some tracks have weird, jumpy behavior before the RGBump
        which gets mistakenly identified as the RGBump. To avoid this, I force the
        RGBump to be the first local minimum in Teff after the TAMS *and* with
        a luminosity above 10 Lsun.

        Added 2019/05/28: The default grid has two tracks that *just barely* do
        not reach the RGBump. These tracks will use _RGBump_special. In this
        function, I manually set the final point in these tracks as the RGBump
        to extend their EEPs. This will only affect calculations pas the TAMS
        for stars adjacent to these tracks in the grid, and the errors should be
        negligible (but I have not quantified them).
        '''
        lum = self.eep_params['lum']
        log_teff = self.eep_params['log_teff']
        N = len(self)

        lum_tr = self.loc[i0:, lum]
        logT_tr = self.loc[i0:, log_teff]

        RGBump = _first_true_index(lum_tr > 10) + 1
        if RGBump == 0:
            return -1

        while logT_tr[RGBump] < logT_tr[RGBump-1] and RGBump < N-1:
            RGBump += 1

        # Two cases: 1) We didn't reach an extremum, in which case RGBump gets
        # set as the final index of the track. In this case, return -1.
        # 2) We found the extremum, in which case RGBump gets set
        # as the index corresponding to the extremum.
        if RGBump >= N-1:
            return -1
        return RGBump-1

    def get_RGBTip(self, i0=None):
        '''
        Red Giant Branch Tip
        Dotter describes the tip of the red giant branch (RGBTip) EEP as
        "the point at which stellar luminosity reaches a maximum---or the stellar
        Teff reaches a minimum---after core H burning is complete but before core
        He burning has progressed significantly."

        Note that the YREC models at the time of this writing nominally end at
        the helium flash, so the RGBTip is unadvisable to use as an EEP.
        '''

        core_helium_frac = self.eep_params['core_helium_frac']
        lum = self.eep_params['lum']
        log_teff = self.eep_params['log_teff']

        Ymin = self.loc[i0, core_helium_frac] - 1e-2
        Yc_tr = self.loc[i0:, core_helium_frac]
        before_He_burned = (Yc_tr > Ymin)
        if not before_He_burned.any():
            return -1

        lum_tr = self.loc[i0:, lum]
        RGBTip1 = (lum_tr[before_He_burned]).idxmax()

        logT_tr = self.loc[i0:, log_teff]
        RGBTip2 = (logT_tr[before_He_burned]).idxmin()

        RGBTip = min(RGBTip1, RGBTip2)
        return RGBTip
    
    def _Metric_Function(self):
        '''
        The Metric Function is used to calculate the distance along the evolution
        track. Traditionally, the Euclidean distance along the track on the
        H-R diagram has been used, but any positive-definite function will work.
        '''
        return self._HRD_distance()


    def _HRD_distance(self):
        '''
        Distance along the H-R diagram, to be used in the Metric Function.
        Returns an array containing the distance from the beginning of the 
        evolution track for each step along the track, in logarithmic effective
        temperature and logarithmic luminosity space.
        '''

        # Allow for scaling to make changes in Teff and L comparable
        Tscale = self.eep_params['teff_scale']
        Lscale = self.eep_params['lum_scale']

        log_teff = self.eep_params['log_teff']
        lum = self.eep_params['lum']

        logTeff = self[log_teff]
        logLum = np.log10(self[lum])

        N = len(self)
        dist = np.zeros(N)
        for i in range(1, N):
            temp_dist = (((logTeff.iloc[i] - logTeff.iloc[i-1])*Tscale)**2
                        + ((logLum.iloc[i] - logLum.iloc[i-1])*Lscale)**2)
            dist[i] = dist[i-1] + np.sqrt(temp_dist)

        return dist

    def get_eep_track_lengths(self):
        if 'eep' not in self.index.names:
            raise KeyError('Grid is wrong kind. Must be EEP grid.')

        idx = self.index.droplevel('eep').drop_duplicates()
        lengths = [len(self.loc[i]) for i in idx]
        lengths = pd.DataFrame(lengths, index=idx)
        return lengths

class StarGridInterpolator(DFInterpolator):
    def __init__(self, grid, eep_params):
        super(StarGridInterpolator, self).__init__(grid)

        self.name = grid.name
        self.columns = grid.columns

        self.mass_lim = grid.get_mass_lim()
        self.met_lim = grid.get_met_lim()
        self.alpha_lim = grid.get_alpha_lim()

        self.max_eep = grid.index.to_frame().eep.max()
        self.eep_params = eep_params

    def get_star_eep(self, mass, met, alpha, eep):
        star_values = self((mass, met, alpha, eep))
        return pd.Series(star_values, index=self.columns)

    def get_star_age(self, mass, met, alpha, age):
        track = self.get_track(mass, met, alpha)
        labels = track.columns
        interpf = interp1d(track[self.eep_params['age']], track.values.T)
        star = pd.Series(interpf(age), labels)
        return star

    def fit_star(self, star_dict,
                 loss='leastsq',
                 guess0=(1, 0, 0, 250), 
                 bounds=[(0.3, 2.0), (-1, 0.5), (0, 0.4), (0, 606)]):
        '''
        Fit a star from data using scipy.optimize.minimize.

        PARAMETERS
        ----------
        star_dict: dict containing label-value pairs for the star to be fit

        guess0: tuple containing initial guess of input values for star. 
            These should be of the same form as the input to 
            StarGridInterpolator.get_star_eep.

        bounds: a sequence of (min, max) tuples for each input parameter.

        RETURNS
        -------
        star: pandas.Series of StarGridInterpolator output for result.

        result: the output of scipy.optimize.minimize.
        '''

        if loss == 'leastsq':
            loss_function = self._leastsq
        elif loss == 'meanpercenterror':
            loss_function = self._meanpcterr

        result = minimize(loss_function, guess0, args=(star_dict,), bounds=bounds)
        star = self.get_star_eep(*result.x)
        return star, result   

    def _leastsq(self, x, star_dict):
        star = self.get_star_eep(*x)
        ssq = np.average(
            [(star[l] - star_dict[l])**2 for l in star_dict]
        )
        return ssq

    def _meanpcterr(self, x, star_dict):
        star = self.get_star_eep(*x)
        mpe = np.average(
                [np.abs(star[l] - star_dict[l])/star_dict[l] for l in star_dict]
        )
        return mpe

    def _test_fit(self):
        sun = {'R/Rsun':1, 'L/Lsun':1, 'Z/X(surf)': 0.02289, 'Age(Gyr)':4.47}
        return self.fit_star(sun, bounds=[(0.9, 1.1), (-0.3, 0.3), (0, 0), (250, 400)])

    def get_track(self, mass, met, alpha, eep=None):
        if eep is None:
            num_eeps = self.max_eep + 1
            ones_arr = np.ones(num_eeps)
            eep = range(num_eeps)
        idx = [mass*ones_arr, met*ones_arr, alpha*ones_arr, np.array(eep)]
        star_values = self(idx)
        return StarGrid(star_values, columns=self.columns, name=self.name)

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


def from_pandas(df, name=None):
    return StarGrid(df, name=name)

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
    df = pd.read_parquet(*args, **kwargs)
    return from_pandas(df, name=name)

def install_grid(script):
    module = import_module(script)

    # Create cache directories
    path = os.path.join(grids_path, module.name)
    if not os.path.exists(path):
        os.makedirs(path)

    eep_params = module.eep_params
    # Cache eep parameters
    with open(os.path.join(path, 'eep_params.pkl'), 'wb') as f: 
        pickle.dump(eep_params, f)
    
    print('Reading and combining grid files')
    grids = module.setup()
    
    full_save_path = os.path.join(path, 'full_grid.pqt')
    print(f'Saving to {full_save_path}')
    grids.to_parquet(full_save_path)

    print(f'Converting to eep-based tracks')
    eeps = grids.to_eep(**eep_params)

    eep_save_path = os.path.join(path, 'eep_grid.pqt')
    print(f'Saving to {eep_save_path}')
    eeps.to_parquet(eep_save_path)

    interp = StarGridInterpolator(eeps, eep_params)
    interp_save_path = os.path.join(path, 'interpolator.pkl')
    print(f'Saving interpolator to {interp_save_path}')
    interp.to_pickle(path=interp_save_path)

    print(f'Model grid "{module.name}" installed.')

def load_full_grid(name):
    return load_grid(name, kind='full')

def load_eep_grid(name):
    return load_grid(name, kind='eep')

def load_grid(name, kind='full'):
    file_path = os.path.join(grids_path, name, f'{kind}_grid.pqt')
    if os.path.exists(file_path):
        return read_parquet(file_path, name=name)
    raise FileNotFoundError(f'{file_path}: No such file exists.')

def load_eep_params(name):
    params_path = os.path.join(grids_path, name, 'eep_params.pkl')
    with open(params_path, 'rb') as f: 
        eep_params = pickle.load(f)

    return eep_params

def _first_true_index(bools):
    '''
    Given a pandas Series of bools, returns the index of the first occurrence
    of `True`. **Index-based, NOT location-based**
    I.e., say x = pd.Series({0: False, 2: False, 4: True}), then
    _first_true_index(x) will return index 4, not the positional index 2.

    If no value in `bools` is True, returns -1.
    '''
    if not bools.any():
        return -1
    bools.idxmax()
    return bools.idxmax()