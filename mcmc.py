import numpy as np
import pandas as pd
import emcee

import kiauhoku as kh


def lnprob(theta, data, sigma, grid):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf, None

    ll, star = lnlike(theta, data, sigma, grid)

    if not np.isfinite(ll):
        return -np.inf, None

    my_log_prob = lp + ll
    star["lnprob"] = my_log_prob

    return my_log_prob, star

def lnprior(theta):
    '''
    With the current StarGridInterpolator implementation,
    there's no need to check if a move is within the boundaries.
    If it's not, it will return all NaNs, which gets caught in the
    likelihood function.
    '''
    return 0
    
    mass, met, alpha, eep = theta
    if (0.3 <= mass  <= 2.0 and
       -1.0 <= met   <= 0.5 and
        0.0 <= alpha <= 0.4 and
        200 <=  eep <=  600): # Between ZAMS and RGB bump
        return 0
    return -np.inf

def lnlike(theta, data, sigma, grid):
    mass, met, alpha, eep = theta   # unpack parameters
    star = grid.get_star_eep(*theta)
    if star.isnull().any():
        return -np.inf, None

    teff = 10**star['Log Teff(K)']
    star['Teff(K)'] = teff

    z_x_surf = np.log10(star['Z/X(surf)']/0.02289)
    star['[Z/X]'] = z_x_surf
    star['[alpha/Z]'] = alpha

    prot = star['Prot(days)']

    theta_ = np.array([teff, z_x_surf, alpha, prot])
    my_log_like = lnChiSq(theta_, data, sigma)
    return my_log_like, star

def lnChiSq(theta, data, sigma):
    return -0.5 * (((theta-data)/sigma)**2).sum()

def run_mcmc(
    data, sigma, grid,
    initial_guess, guess_width,
    n_walkers=12, n_burnin=0, n_iter=500,
    save_path=None
):

    pos0 = np.array([
        np.random.normal(initial_guess[l], guess_width[l], n_walkers)
        for l in initial_guess
    ]).T

    sampler = emcee.EnsembleSampler(n_walkers, len(initial_guess),
        log_prob_fn=lnprob, 
        args=(sun_data, sun_sigma, grid),
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


if __name__ == '__main__':
    sun_data = np.array([5776, 0, 0, 24.5])
    sun_sigma = np.array([100, 0.1, 0.1, 2.5])

    initial_guess = {
        'initial mass': 1,
        'initial [M/H]': 0,
        'initial [alpha/M]': 0.2, 
        'eep': 300
    }

    guess_width = {
        'initial mass': 0.2,
        'initial [M/H]': 0.1,
        'initial [alpha/M]': 0.05,
        'eep': 20
    }

    n_walkers = 12
    n_burnin = 100
    n_iter = 500

    grid = kh.load_interpolator('fastlaunch')

    sampler, chains = run_mcmc(
        sun_data, sun_sigma, grid,
        initial_guess, guess_width,
        n_walkers, n_burnin, n_iter,
    )