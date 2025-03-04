from warnings import warn
import numpy as np
from numpy.polynomial.polynomial import polyval
import pandas as pd

'''
This module calculates the habitable zone (HZ) and continuous habitable zone (CHZ)
boundaries for each stellar model evolutionary phase. This is accomplished using
two main functions, add_HZ and add_HZ_custom. Both functions accept a StarGrid
object as input which contains a stellar model grid converted to EEP basis.
add_HZ uses various default HZ prescriptions to calculate the HZ and CHZ, while
add_HZ_custom accepts coefficients as an input to calculate a custom defined HZ.
'''

# Use default HZ prescriptions
def add_HZ(grid,source='K14',which=2,simple=False,wcl=False,chz=True,hzl=2):
	'''
    Parameters
    ----------
    grid (StarGrid object): EEP-based stellar model grid stored as a StarGrid object.

    source (str): Source to use for HZ prescription. Options are:

    			  K93 - Kasting et al. 1993
    			  K13 - Kopparapu et al. 2013
    			  K14 - Kopparapu et al. 2014
    			  W17 - Wolf et al. 2017
    			  R18 - Ramirez & Kaltenegger 2018
	
	which (int): For the selected source, which HZ prescription to use. Options, if
				 available, are:

				 1 - Conservative HZ, usually moist greenhouse IHZ and maximum
				 	 greenhouse OHZ
				 2 - Optimistic HZ, usually runaway greenhouse IHZ and maximum
				 	 greenhouse OHZ
				 3 - Empirical HZ, recent Venus IHZ and early Mars OHZ

	simple (bool): If True, scale HZ boundaries by luminosity only, neglecting
				   dependence on Teff.

	wcl (bool): If True, include Turbet et al. 2023 water condensation limit in HZ
				calculation.

	chz (bool): If True, calculate CHZ evolution.

	hzl (scalar): Habitable zone lifetime, in Gyrs, to use when calculating CHZ.

	Returns
    -------
    grid (StarGrid): grid of EEP-based evolution tracks with added columns for HZ
    				 and CHZ boundaries.

    '''

	# Check if grid is a StarGrid object. More robust way?
	try:
		gname = grid.name
	except:
		raise TypeError('\'grid\' must be StarGrid object. \
			If you would like to use a custom model grid, \
			please install first using the \'custom_install.py\' script.')

	# Get Teff, luminosity, and ZAMS EEP for grid.
	# MIST doesn't have eep_params, so set manually.
	if gname=='mist':
		lum = 10**grid['log_L']
		teff = 10**grid['log_Teff']
		zams = 200
	else:
		lumstr = grid.eep_params['lum']
		lum = grid[lumstr]
		if (lum<0).any():
			lum = 10**lum

		teffstr = grid.eep_params['log_teff']
		teff = 10**grid[teffstr]

		zams = grid.eep_params['intervals'][0]+1

	# Get source HZ parameterization.
	if source in ('K93','k93'):
		if which not in (1,2,3):
			raise ValueError('\'which\' must be set to 1, 2, or 3 for Kasting et al. 1993.')
		else:
			Tstar, c1, c2, Trange = K93(teff,which)

	elif source in ('K13','k13'):
		if which not in (1,2,3):
			raise ValueError('\'which\' must be set to 1, 2, or 3 for Kopparapu et al. 2013.')
		else:
			Tstar, c1, c2, Trange = K13(teff,which)

	elif source in ('K14','k14'):
		if which not in (2,3):
			raise ValueError('\'which\' must be set to 2 or 3 for Kopparapu et al. 2014.')
		else:
			Tstar, c1, c2, Trange = K14(teff,which)

	elif source in ('W17','w17'):
		warn('Only one HZ prescription is available for Wolf et al. 2017. Continuing...')
		Tstar, c1, c2, Trange = W17(teff)

	elif source in ('R18','r18'):
		if which not in (2,3):
			raise ValueError('\'which\' must be set to 2 or 3 for Ramirez and Kaltenegger 2018.')
		else:
			Tstar, c1, c2, Trange = R18(teff,which)

	else:
		raise ValueError('\'source\' value not recognized. Options are K93, K13, K14, W17, and R18.')

	# Calculate HZ distances, check if only luminosity scaling wanted.
	if simple:
		ihz, ohz = calc_HZ_simple(lum,c1[0],c2[0])
	else:
		ihz, ohz = calc_HZ(Tstar,lum,c1,c2,Trange)

	if wcl:
		ihz = T23(teff,lum,ihz,zams,simple)

	# Join IHZ and OHZ with grid
	grid[['IHZ ('+source+'-'+str(which)+')','OHZ ('+source+'-'+str(which)+')']] = pd.concat([ihz,ohz],axis=1)

	# Check if CHZ wanted
	if chz==True:
		# Rescale habitable zone lifetime to grid age scale
		if gname == 'mist':
			hzl = 1e9*hzl
			agestr = 'star_age'
		else:
			agestr = grid.eep_params['age']

		if 'My' in agestr:
			hzl = 1e3*hzl
		elif 'yrs' in agestr:
			hzl = 1e9*hzl

		hz_grid = grid.loc[:, [agestr,grid.columns[-2],grid.columns[-1]]]

		# Calculate CHZ
		chz = calc_CHZ(hz_grid,hzl,zams)

		# Join CHZ with grid
		grid[['ICHZ ('+source+'-'+str(which)+')', 'OCHZ ('+source+'-'+str(which)+')']] = chz

	return grid

# Use custom HZ prescription
def add_HZ_custom(grid,inner,outer,Trange=None,Tref=None,wcl=False,chz=True,hzl=2):
	'''
    Parameters
    ----------
    grid (StarGrid object): EEP-based stellar model grid stored as a StarGrid object.

    inner (scalar or 1D array): Scalar or array of polynomial coefficients defining
    							IHZ prescription in terms of stellar effective flux
    							relative to Earth (Seff). Both inner and outer must
    							be the same type.

    outer (scalar or 1D array): Scalar or array of polynomial coefficients defining
    							OHZ prescription in terms of stellar effective flux
    							relative to Earth (Seff). Both inner and outer must
    							be the same type.

	Trange (tuple or 1D array): Range of Teff values that HZ prescription is defined
								for (i.e. [2600,7200]).

	Tref (scalar): Reference Teff for HZ prescription that defines the polynomial
				   intercept (i.e. this is 5780 K for K14).

	wcl (bool): If True, include Turbet et al. 2023 water condensation limit in HZ
				calculation.

	chz (bool): If True, calculate CHZ evolution.

	hzl (scalar): Habitable zone lifetime, in Gyrs, to use when calculating CHZ.

	Returns
    -------
    grid (StarGrid): grid of EEP-based evolution tracks with added columns for HZ
    				 and CHZ boundaries.

    '''

	# Check if grid is a StarGrid object. More robust way?
	try:
		gname = grid.name
	except:
		raise TypeError('\'grid\' must be StarGrid object. \
			If you would like to use a custom model grid, \
			please install first using the \'custom_install.py\' script.')

	# Get Teff, luminosity, and ZAMS EEP for grid.
	# MIST doesn't have eep_params, so set manually.
	if gname=='mist':
		lum = 10**grid['log_L']
		teff = 10**grid['log_Teff']
		zams = 200
	else:
		lumstr = grid.eep_params['lum']
		lum = grid[lumstr]
		if (lum<0).any():
			lum = 10**lum

		teffstr = grid.eep_params['log_teff']
		teff = 10**grid[teffstr]

		zams = grid.eep_params['intervals'][0]+1

	# Set coefficient variables, make scalar inputs the same type
	c1 = np.float64(inner)
	c2 = np.float64(outer)

	# Check if coefficient variables are both scalars or arrays
	if isinstance(c1,np.float64)!=isinstance(c2,np.float64):
		raise TypeError('\'inner\' and \'outer\' must both be scalars or 1D arrays.')

	# If scalar, do luminosity scaling only
	simple = False
	if isinstance(c1,np.float64):
		simple = True

	# Check if Trange is a tuple or 1D array of length 2
	if simple==False and Trange==None:
		raise ValueError('\'Trange\' must be input for Teff scaling. Input scalars for \'inner\' and \'outer\' to use only luminosity scaling.')

	if Tref is None:
		Tstar = teff
	else:
		Tstar = teff - Tref
		Trange = np.array(Trange) - Tref

	# Calculate HZ distances, check if only luminosity scaling wanted.
	if simple:
		warn('\'inner\' and \'outer\' are scalars, using luminosity scaling only.')
		ihz, ohz = calc_HZ_simple(lum,c1,c2)
	else:
		ihz, ohz = calc_HZ(Tstar,lum,c1,c2,Trange)

	if wcl:
		ihz = T23(teff,lum,ihz,zams,simple)

	# Join IHZ and OHZ with grid
	grid[['IHZ','OHZ']] = pd.concat([ihz,ohz],axis=1)

	# Check if CHZ wanted
	if chz==True:
		# Rescale habitable zone lifetime to grid age scale
		if gname == 'mist':
			hzl = 1e9*hzl
			agestr = 'star_age'
		else:
			agestr = grid.eep_params['age']

		if 'My' in agestr:
			hzl = 1e3*hzl
		elif 'yrs' in agestr:
			hzl = 1e9*hzl

		hz_grid = grid.loc[:, [agestr,grid.columns[-2],grid.columns[-1]]]

		# Calculate CHZ
		chz = calc_CHZ(hz_grid,hzl,zams)

		# Join CHZ with grid
		grid[['ICHZ','OCHZ']] = chz

	return grid

# Calc HZ using Teff scaling
def calc_HZ(Tstar,lum,c1,c2,Trange):
	'''
    Parameters
    ----------
    Tstar (Series): Multiindex series containing Teff column from model grid shifted
    				by Tref.

    lum (Series): Multiindex series containing luminosity column from model grid.

    c1 (1D array): Array of polynomial coefficients defining IHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

    c2 (1D array): Array of polynomial coefficients defining OHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

	Trange (tuple or 1D array): Range of Teff values that HZ prescription is defined
								for, shifted by Tref.

	Returns
    -------
    ihz (Series): Multiindex series containing IHZ boundary at each EEP.

    ohz (Series): Multiindex series containing OHZ boundary at each EEP.

    '''

	# Scale by Teff and luminosity inside Teff range, else scale by luminosity only 
	seff1 = Tstar.apply(lambda x: polyval(x, c1) if Trange[0]<=x<=Trange[1] else (polyval(Trange[0], c1) if x<Trange[0] else polyval(Trange[1], c1)))
	seff2 = Tstar.apply(lambda x: polyval(x, c2) if Trange[0]<=x<=Trange[1] else (polyval(Trange[0], c2) if x<Trange[0] else polyval(Trange[1], c2)))

	ihz = (lum/seff1)**0.5
	ohz = (lum/seff2)**0.5

	return ihz, ohz

# Calc HZ using luminosity scaling
def calc_HZ_simple(lum,c1,c2):
	'''
    Parameters
    ----------
    lum (Series): Multiindex series containing luminosity column from model grid.

    c1 (scalar): Scalar defining IHZ in terms of stellar effective flux relative
    			 to Earth (Seff).

    c2 (scalar): Scalar defining OHZ in terms of stellar effective flux relative
    			 to Earth (Seff).

	Returns
    -------
    ihz (Series): Multiindex series containing IHZ boundary at each EEP.

    ohz (Series): Multiindex series containing OHZ boundary at each EEP.

    '''

	ihz = (lum/c1)**0.5
	ohz = (lum/c2)**0.5

	# Leftover column name causes series to be treated as dataframe
	ihz.name = None
	ohz.name = None

	return ihz, ohz

# Calc CHZ for input habitable zone time limit
def calc_CHZ(hz_grid,hzl,zams):
	'''
    Parameters
    ----------
    hz_grid (series): Multiindex dataframe containing age, IHZ, and OHZ for EEP-based
    				  grid.

    hzl (scalar): Habitable zone lifetime, in Gyrs, to use when calculating CHZ.

    zams (int): EEP value for ZAMS.

	Returns
    -------
    chz (dataframe): Multiindex dataframe containing CHZ boundaries at each EEP.

    '''

	chz = pd.DataFrame(columns=['ICHZ', 'OCHZ'], index=hz_grid.index)

	# Group by all index levels except for the 'eep' level for iteration by model
	hz_grid = hz_grid.groupby(hz_grid.index.droplevel('eep'))

	# Iterate by model and calculate CHZ boundaries at each timestep
	for model, model_df in hz_grid:
		eeps = model_df.index.get_level_values('eep')
		ages = model_df.iloc[:, 0].values
		ihz = model_df.iloc[:, 1].values
		ohz = model_df.iloc[:, 2].values

		ichz = []
		ochz = []

		# Track ICHZ to ensure it never decreases
		ichz_max = 0

		# Start age (ZAMS + HZL)
		age_start = ages[zams] + hzl

		# Iterate over each eep and calculate CHZ
		for i in eeps:
			age_i = ages[i]

			# If the timestep is old enough, calculate the CHZ
			if age_i >= age_start:
				# Get the distances for the current timestep and previous valid timesteps
				mp = 1e-8 * age_i # Account for machine precision

				period = (ages >= age_i - hzl) & (ages <= age_i + mp)

				ichz_i = ihz[period].max()  # Max of the IHZ over time period
				ochz_i = ohz[period].min()  # Min of the OHZ over time period

				# Check if CHZ exists
				if ichz_i < ochz_i:
					# Ensure the ICHZ does not decrease
					if ichz_i >= ichz_max:
						ichz.append(ichz_i)
						ichz_last = ichz_i

					else:
						ichz.append(ichz_max) # Retain the max ICHZ

					ochz.append(ochz_i)

				# CHZ doesn't exist
				else:
					ichz.append(np.nan)
					ochz.append(np.nan)

			# Not enough time for CHZ
			else:
				ichz.append(np.nan)
				ochz.append(np.nan)

		# Add model CHZ to dataframe
		chz.loc[model_df.index, 'ICHZ'] = ichz
		chz.loc[model_df.index, 'OCHZ'] = ochz

	return chz

# Kasting et al. 1993
def K93(teff,which):
	'''
    Parameters
    ----------
    teff (Series): Multiindex series containing Teff column from model grid.

    which (int): HZ prescription to use. Options are:

				 1 - Conservative HZ, moist greenhouse IHZ and maximum greenhouse OHZ
				 2 - Optimistic HZ, runaway greenhouse IHZ and maximum greenhouse OHZ
				 3 - Empirical HZ, recent Venus IHZ and early Mars OHZ

	Returns
    -------
	Tstar (Series): Multiindex series containing Teff column from model grid shifted
    				by Tref.

    c1 (1D array): Array of polynomial coefficients defining IHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

    c2 (1D array): Array of polynomial coefficients defining OHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

	Trange (tuple or 1D array): Range of Teff values that HZ prescription is defined
								for, shifted by Tref.

    '''

	# Range = 3700 - 7200 K
	Tlo = 3700
	Thi = 7200

	# Reference Teff = 5700
	Tref = 5700
	Tstar = teff - Tref
	Tstarlo = Tlo - Tref
	Tstarhi = Thi - Tref
	Trange = (Tstarlo,Tstarhi)

	# Conservative: water loss and maximum greenhouse
	if which==1:
		# Water loss
		c1 = [1.1,7.85714286e-05,1.42857143e-08]

		# Maximum greenhouse
		c2 = [0.36,5.73809524e-05,6.19047619e-09]

	# Optimistic: runaway greenhouse and maximum greenhouse
	elif which==2:
		# Runaway greenhouse
		c1 = [1.41,2.63809524e-04,4.19047619e-08]

		# Maximum greenhouse
		c2 = [0.36,5.73809524e-05,6.19047619e-09]

	# Empirical: Recent Venus and Early Mars
	elif which==3:
		# Recent Venus
		c1 = [1.76,1.25714286e-04,2.28571429e-08]

		# Early Mars
		c2 = [0.32,5.14285714e-05,5.71428571e-09]

	return Tstar, c1, c2, Trange

# Kopparapu et al. 2013
def K13(teff,which):
	'''
    Parameters
    ----------
    teff (Series): Multiindex series containing Teff column from model grid.

    which (int): HZ prescription to use. Options are:

				 1 - Conservative HZ, moist greenhouse IHZ and maximum greenhouse OHZ
				 2 - Optimistic HZ, runaway greenhouse IHZ and maximum greenhouse OHZ
				 3 - Empirical HZ, recent Venus IHZ and early Mars OHZ

	Returns
    -------
	Tstar (Series): Multiindex series containing Teff column from model grid shifted
    				by Tref.
    
    c1 (1D array): Array of polynomial coefficients defining IHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

    c2 (1D array): Array of polynomial coefficients defining OHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

	Trange (tuple or 1D array): Range of Teff values that HZ prescription is defined
								for, shifted by Tref.

    '''

	# Range = 2600 - 7200 K
	Tlo = 2600
	Thi = 7200

	# Reference Teff = 5780
	Tref = 5780
	Tstar = teff - Tref
	Tstarlo = Tlo - Tref
	Tstarhi = Thi - Tref
	Trange = (Tstarlo,Tstarhi)

	# Conservative: water loss and maximum greenhouse
	if which==1:
		# Water loss
		c1 = [1.0146,8.1884e-5,1.9394e-9,-4.3618e-12,-6.8260e-16]

		# Maximum greenhouse
		c2 = [0.3507,5.9578e-5,1.6707e-9,-3.0058e-12,-5.1925e-16]

	# Optimistic: runaway greenhouse and maximum greenhouse
	elif which==2:
		# Runaway greenhouse
		c1 = [1.0385,1.2456e-4,1.4612e-8,-7.6345e-12,-1.7511e-15]

		# Maximum greenhouse
		c2 = [0.3507,5.9578e-5,1.6707e-9,-3.0058e-12,-5.1925e-16]

	# Empirical: Recent Venus and Early Mars
	elif which==3:
		# Recent Venus
		c1 = [1.7763,1.4335e-4,3.3954e-9,-7.6364e-12,-1.1950e-15]

		# Early Mars
		c2 = [0.3207,5.4471e-5,1.5275e-9,-2.1709e-12,-3.8282e-16]

	return Tstar, c1, c2, Trange

# Kopparapu et al. 2014
def K14(teff,which):
	'''
    Parameters
    ----------
    teff (Series): Multiindex series containing Teff column from model grid.

    which (int): HZ prescription to use. Options are:

				 2 - Optimistic HZ, runaway greenhouse IHZ and maximum greenhouse OHZ
				 3 - Empirical HZ, recent Venus IHZ and early Mars OHZ

	Returns
    -------
	Tstar (Series): Multiindex series containing Teff column from model grid shifted
    				by Tref.
    
    c1 (1D array): Array of polynomial coefficients defining IHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

    c2 (1D array): Array of polynomial coefficients defining OHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

	Trange (tuple or 1D array): Range of Teff values that HZ prescription is defined
								for, shifted by Tref.

    '''

	# Range = 2600 - 7200 K
	Tlo = 2600
	Thi = 7200

	# Reference Teff = 5780
	Tref = 5780
	Tstar = teff - Tref
	Tstarlo = Tlo - Tref
	Tstarhi = Thi - Tref
	Trange = (Tstarlo,Tstarhi)

	# Optimistic: runaway greenhouse and maximum greenhouse
	if which==2:
		# Runaway greenhouse
		c1 = [1.107,1.332e-4,1.58e-8,-8.308e-12,-1.931e-15]

		# Maximum greenhouse
		c2 = [0.356,6.171e-5,1.698e-9,-3.198e-12,-5.575e-16]

	# Empirical: Recent Venus and Early Mars
	elif which==3:
		# Recent Venus
		c1 = [1.776,2.136e-4,2.533e-8,-1.332e-11,-3.097e-15]

		# Early Mars
		c2 = [0.32,5.547e-5,1.526e-9,-2.874e-12,-5.011e-16]

	# Runaway greenhouse 5 M_e

	#[1.188 1.433e-4 1.707e-8 -8.968e-12 -2.084e-15]

	# Runaway greenhouse 0.1 M_e

	#[0.99 1.209e-4 1.404e-8 -7.418e-12 -1.713e-15]

	return Tstar, c1, c2, Trange

# Wolf et al. 2017
def W17(teff):
	'''
    Parameters
    ----------
    teff (Series): Multiindex series containing Teff column from model grid.

	Returns
    -------
	Tstar (Series): Multiindex series containing Teff column from model grid shifted
    				by Tref.
    
    c1 (1D array): Array of polynomial coefficients defining IHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

    c2 (1D array): Array of polynomial coefficients defining OHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

	Trange (tuple or 1D array): Range of Teff values that HZ prescription is defined
								for, shifted by Tref.

    '''

	# Range = 4900 - 6600 K
	Tlo = 4900
	Thi = 6600

	# Reference Teff = 5780
	Tref = 5780
	Tstar = teff - Tref
	Tstarlo = Tlo - Tref
	Tstarhi = Thi - Tref
	Trange = (Tstarlo,Tstarhi)

	# Water loss 1 Gyr
	c1 = [1.19645,1.39815e-4,3.12706e-8]

	# Snowball
	c2 = [0.92515,7.27318e-5,9.82310e-10]

	return Tstar, c1, c2, Trange

# Ramirez & Kaltenegger 2018
def R18(teff,which):
	'''
    Parameters
    ----------
    teff (Series): Multiindex series containing Teff column from model grid.

    which (int): HZ prescription to use. Options are:

				 2 - Optimistic HZ, runaway greenhouse IHZ and maximum greenhouse OHZ
				 3 - Empirical HZ, recent Venus IHZ and early Mars OHZ

	Returns
    -------
	Tstar (Series): Multiindex series containing Teff column from model grid shifted
    				by Tref.
    
    c1 (1D array): Array of polynomial coefficients defining IHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

    c2 (1D array): Array of polynomial coefficients defining OHZ prescription in terms
    			   of stellar effective flux relative to Earth (Seff).

	Trange (tuple or 1D array): Range of Teff values that HZ prescription is defined
								for, shifted by Tref.

    '''

	# Range = 2600 - 10000 K
	Tlo = 2600
	Thi = 10000

	# Reference Teff = 5780
	Tref = 5780
	Tstar = teff - Tref
	Tstarlo = Tlo - Tref
	Tstarhi = Thi - Tref
	Trange = (Tstarlo,Tstarhi)

	# Optimistic: runaway greenhouse and maximum greenhouse
	if which==2:
		# Runaway greenhouse
		c1 = [1.105,1.1921e-4,9.5932e-9,-2.6189e-12,1.3710e-16]

		# Maximum greenhouse
		c2 = [0.3587,5.8087e-5,1.5393e-9,-8.3547e-13,1.0319e-16]

	# Empirical: Recent Venus and Early Mars
	elif which==3:
		# Recent Venus
		c1 = [1.768,1.3151e-4,5.8695e-10,-2.8895e-12,3.2174e-16]

		# Early Mars
		c2 = [0.3246,5.213e-5,4.5245e-10,1.0223e-12,9.6376e-17]

	return Tstar, c1, c2, Trange

# Turbet et al. 2023
def T23(teff,lum,ihz,zams,simple):
	'''
    Parameters
    ----------
    teff (Series): Multiindex series containing Teff column from model grid.

    lum (Series): Multiindex series containing luminosity column from model grid.

    ihz (Series): Multiindex series containing IHZ boundary at each EEP.

    zams (int): EEP value for ZAMS.

    simple (bool): If True, scale HZ boundaries by luminosity only, neglecting
				   dependence on Teff.

	Returns
    -------
	ihz (Series): Multiindex series containing IHZ boundary at each EEP, taking
				  into account water condensation limit.

    '''

	# Range = 2600 - 7200 K
	Tlo = 2600
	Thi = 7200

	# Reference Teff = 5780
	Tref = 5780
	Tstar = teff - Tref
	Tstarlo = Tlo - Tref
	Tstarhi = Thi - Tref

	# Water condensation limit (K13 maximum greenhouse Seff - 0.065)
	c = [0.9496,8.1884e-5,1.9394e-9,-4.3618e-12,-6.8260e-16]

	# Calculate effective flux
	# Scale by luminosity only
	if simple:
		wcl = (lum/c[0])**0.5

	# Scale by Teff and luminosity
	else:
		seff = c[0] + c[1]*Tstar + c[2]*Tstar**2 + c[3]*Tstar**3 + c[4]*Tstar**4

		# Scale by luminosity only outside Teff range
		seff[teff<Tlo] = c[0] + c[1]*Tstarlo + c[2]*Tstarlo**2 + c[3]*Tstarlo**3 + c[4]*Tstarlo**4
		seff[teff>Thi] = c[0] + c[1]*Tstarhi + c[2]*Tstarhi**2 + c[3]*Tstarhi**3 + c[4]*Tstarhi**4

		wcl = (lum/seff)**0.5

	# Stop evolving WCL at ZAMS
	zamsv = wcl.xs(zams, level='eep')

	# MS indices
	msi = ihz.index.get_level_values('eep') > zams

	wcl.loc[msi] = zamsv

	# Set IHZ to WCL where distance is greater
	ihz.loc[ihz<wcl] = wcl.loc[ihz<wcl]

	return ihz



