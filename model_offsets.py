import numpy as np
import matplotlib.pyplot as plt
import kiauhoku as kh

# Solar surface metallicity Z/X from Grevesse & Sauval 1998.
solar_z_x = 0.02289

yrec = kh.load_interpolator('yrec')
mist = kh.load_interpolator('mist')
dart = kh.load_interpolator('dartmouth')
gars = kh.load_interpolator('garstec')

def yrec_prob(theta, grid, data, error):
    model = grid.get_star_eep(theta)
    if model.isnull().any():
        return -np.inf, None

    model_teff = 10**model['Log Teff(K)']
    model_lum = 10**model['L/Lsun']
    model_met = np.log10(model['Zsurf']/model['Xsurf']/solar_z_x)

    model_params = np.array([model_teff, model_lum, model_met])
    data = np.array(data)
    error = np.array(error)

    chisq = -0.5 * np.sum(((model_params - data)/error)**2)

    model['teff'] = model_teff
    model['luminosity'] = model_lum
    model['metallicity'] = model_met
    model['age'] = model['Age(Gyr)']
    model['lnprob'] = chisq
    
    return chisq, model

def mist_prob(theta, grid, data, error):
    model = grid.get_star_eep(theta)
    if model.isnull().any():
        return -np.inf, None
    
    model_teff = 10**model['log_Teff']
    model_lum = 10**model['log_L']
    model_met = model['log_surf_z'] - np.log10(model['surface_h1']*solar_z_x)

    model_params = np.array([model_teff, model_lum, model_met])
    data = np.array(data)
    error = np.array(error)

    chisq = -0.5 * np.sum(((model_params - data)/error)**2)

    model['teff'] = model_teff
    model['luminosity'] = model_lum
    model['metallicity'] = model_met
    model['age'] = model['star_age'] / 1e9
    model['lnprob'] = chisq

    return chisq, model

def dart_prob(theta, grid, data, error):
    model = grid.get_star_eep(theta)
    if model.isnull().any():
        return -np.inf, None
    
    model_teff = 10**model['Log T']
    model_lum = 10**model['Log L']
    model_met = np.log10(model['(Z/X)_surf']/solar_z_x)

    model_params = np.array([model_teff, model_lum, model_met])
    data = np.array(data)
    error = np.array(error)

    chisq = -0.5 * np.sum(((model_params - data)/error)**2)

    model['teff'] = model_teff
    model['luminosity'] = model_lum
    model['metallicity'] = model_met
    model['age'] = model['Age (yrs)'] / 1e9
    model['lnprob'] = chisq

    return chisq, model

def gars_prob(theta, grid, data, error):
    model = grid.get_star_eep(theta)
    if model.isnull().any():
        return -np.inf, None
    
    model_teff = model['Teff']
    model_lum = 10**model['Log L/Lsun']
    model_met = np.log10(model['Zsurf']/model['Xsurf']/solar_z_x)

    model_params = np.array([model_teff, model_lum, model_met])
    data = np.array(data)
    error = np.array(error)

    chisq = -0.5 * np.sum(((model_params - data)/error)**2)

    model['teff'] = model_teff
    model['luminosity'] = model_lum
    model['metallicity'] = model_met
    model['age'] = model['Age(Myr)'] / 1e3
    model['lnprob'] = chisq

    return chisq, model

def mcmc(grid, lnprob, data, err):
    sampler, chains = grid.mcmc_star(
        lnprob,
        args=(data, err),
        initial_guess=(1, 0, 300),
        guess_width=(0.1, 0.1, 25),
        n_walkers=12,
        n_burnin=100,
        n_iter=10000
    )
    
    return sampler, chains

def plot(xlabel, ylabel, fmts, labels, *args, **kw):
    fig, ax = plt.subplots()

    ax.plot(xlabel, ylabel, fmts[0], data=yrec_chains, label=labels[0])
    ax.plot(xlabel, ylabel, fmts[1], data=mist_chains, label=labels[1])
    ax.plot(xlabel, ylabel, fmts[2], data=dart_chains, label=labels[2])
    ax.plot(xlabel, ylabel, fmts[3], data=gars_chains, label=labels[3])

    ax.set(xlabel=xlabel, ylabel=ylabel)

    return fig, ax

def compute_offsets(label, ref='yrec'):
    if ref == 'yrec':
        ref_val = yrec_chains[label].median()
    elif ref == 'mist':
        ref_val = mist_chains[label].median()
    elif ref == 'dart':
        ref_val = dart_chains[label].median()
    elif ref == 'gars':
        ref_val = gars_chains[label].median()

    offsets = {
        f'yrec-{ref}': yrec_chains[label].median() - ref_val,
        f'mist-{ref}': mist_chains[label].median() - ref_val,
        f'dart-{ref}': dart_chains[label].median() - ref_val,
        f'gars-{ref}': gars_chains[label].median() - ref_val
    }

    offsets.pop(f'{ref}-{ref}')
    return offsets

teff, teff_err = input('Enter Teff, err: ').split(',')
lum, lum_err = input('Enter Lum, err: ').split(',')
met, met_err = input('Enter Met, err: ').split(',')

data = [float(teff), float(lum), float(met)]
err = [float(teff_err), float(lum_err), float(met_err)]

yrec_sampler, yrec_chains = mcmc(yrec, yrec_prob, data, err)
mist_sampler, mist_chains = mcmc(mist, mist_prob, data, err)
dart_sampler, dart_chains = mcmc(dart, dart_prob, data, err)
gars_sampler, gars_chains = mcmc(gars, gars_prob, data, err)

print(compute_offsets('age'))
print(compute_offsets('initial_mass'))