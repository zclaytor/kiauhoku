# [Kīauhōkū][kiauhoku github]

[![ascl:2011.027](https://img.shields.io/badge/ascl-2011.027-blue.svg?colorB=262255)](https://ascl.net/2011.027)
[![GitHub version](https://badge.fury.io/gh/zclaytor%2Fkiauhoku.svg)](https://badge.fury.io/gh/zclaytor%2Fkiauhoku)
[![PyPI version](https://badge.fury.io/py/kiauhoku.svg)](https://badge.fury.io/py/kiauhoku)

Python utilities for stellar model grid interpolation.

If you find this package useful, please cite [Claytor et al. (2020)][gyro paper].

Download the model grids from [Zenodo][zenodo].

(C) [Zachary R. Claytor][zclaytor]  
Institute for Astronomy  
University of Hawaiʻi  
2021 July 13

Kīauhōkū  
From Hawaiian:  
1. vt. To sense the span of a star's existence (i.e., its age).  
2. n. The speed of a star (in this case, its rotational speed).  

This name was created in partnership with Dr. Larry Kimura and Bruce Torres Fischer, a student participant in *A Hua He Inoa*, a program to bring Hawaiian naming practices to new astronomical discoveries. We are grateful for their collaboration.

Kīauhōkū is a suite of Python tools to interact with, manipulate, and interpolate between stellar evolutionary tracks in a model grid. It was designed to work with the model grid used in [Claytor et al. (2020)][gyro paper], which was generated using YREC with the magnetic braking law of [van Saders et al. (2013)][van Saders], but other stellar evolution model grids are available. 

## Installation

Kīauhōkū requires the use of Python 3 and uses the following Python packages:  
- numpy  
- scipy  
- pandas  
- matplotlib  
- miniutils
- pyarrow (or some package that supports parquet files)
- numba
- [emcee][emcee]

Personally, I create a conda environment for this. In this example I'll call it "stars".
```bash
conda create -n stars numpy scipy pandas matplotlib pyarrow numba emcee
conda activate stars
pip install git+https://github.com/zclaytor/kiauhoku
```

Brand new: Kīauhōkū is on PyPI! It requires Python 3, but you can finally do this:
```bash
pip install kiauhoku
```
You still need to download the grids from Zenodo and follow grid-specific install instructions. I'm working on including the basic files with the pip install.

## I don't care about the documentation. Just let me get started!
1. Download the model grids from [Zenodo][zenodo].

2. Open an interactive Python session in the directory with the install script. Here we'll install the "fastlaunch" grid, which is a YREC grid that's been processed with the Rotevol rotation tracer code:
```python
from kiauhoku import rotevol
path_to_grid = wherever_you_installed_grids + '/grids/fastlaunch'
rotevol.install(path_to_grid)
```

3. You're ready to go! You can import and interpolate away.
```python
import kiauhoku as kh
grid = kh.load_interpolator('fastlaunch')
```

## How it works

We start with output evolution tracks from your favorite stellar modeling software. For `rotevol` output, these are the \*.out files. Each \*.out file has, for one specific initial metallicity and alpha-abundance, a series of evolution tracks for a range of initial masses. The "fastlaunch" grid for `kiauhoku` has eight \*.out files, corresponding to  
[M/H] ~ [-1.0, -0.5, 0.0, 0.5] and  
[alpha/M] ~ [0.0, 0.4].  
Each file contains 171 evolution tracks for 0.30 <= M/Msun <= 2.00 in steps of 0.01\*Msun.

1. First we load the tracks into a pandas MultiIndexed DataFrame and save to a parquet file.

2. Age is not an optimal dimension for comparing consecutive evolution tracks. For this reason we condense each evolution track in the time domain to a series of Equivalent Evolutionary Phases (EEPs) after the method of Dotter (2016). The EEP-based tracks are packaged into a MultiIndexed DataFrame and saved to parquet.

3. We finally load the EEP-based tracks into a `kiauhoku.stargrid.StarGridInterpolator` object. The `StarGridInterpolator` is based on the DataFrameInterpolator (`DFInterpolator`) from Tim Morton's [`isochrones`][isochrones] package. It performs linear interpolation between consecutive evolution tracks for an input mass, metallicity, alpha-abundance, and either age or EEP-index. We then pickle the interpolator so it can be accessed quickly and easily.


## Basic Usage

Once you have everything running, try doing this:  
```python
import kiauhoku as kh
grid = kh.load_interpolator('fastlaunch')
star = grid.get_star_eep((1, 0, 0, 330))
```

If it works, you should get something close to the sun. The argument to get_star_eep is a tuple containing the model grid indices. In this case, those are mass (in solar units), metallicity, alpha-abundance, and EEP index. See the documentation for more details.

Kīauhōkū comes with MCMC functionality through `emcee`. See the jupyter notebook `mcmc.ipynb` for an example.

   
## Installing Custom Model Grids

To install your own custom grid, you will want to create a setup script (see `custom_install.py` for an example). The only requirements are that your setup file contains (1) a function called `setup` that returns a pandas MultiIndexed DataFrame containing all your evolution tracks, (2) a variable `name` that is set to whatever you want your installed grid to be named, and (3) a variable `raw_grids_path` that sets the path to wherever your custom raw grid is downloaded.

The index for this DataFrame is what all the "get" functions will use to get and interpolate tracks and EEPs. Thus, if you want to access your grid using mass and metallicity, you'll want the DataFrame returned by `setup` to have mass and metallicity, as well as a column to represent the time/EEP step.

You can also use the setup file to define custom EEP functions (see `custom_install.my_RGBump`) for an example) and to tell `kiauhoku` which columns to use in its default EEP functions.

Once your setup file is ready, you can install your custom grid using
```python
import kiauhoku as kh
kh.install_grid('custom_install')
```

If you create a setup file for your favorite model grid and you'd like it to be public, create a pull request and I'll add you as a contributor!


[kiauhoku github]: https://github.com/zclaytor/kiauhoku
[zclaytor]: https://zclaytor.github.io
[gyro paper]: https://ui.adsabs.harvard.edu/abs/2020ApJ...888...43C/abstract
[van Saders]: https://ui.adsabs.harvard.edu/abs/2013ApJ...776...67V/abstract
[emcee]: https://emcee.readthedocs.io/en/latest/
[isochrones]: https://isochrones.readthedocs.io/en/latest/
[zenodo]: https://doi.org/10.5281/zenodo.4287717
