# [Kīauhōkū][kiauhoku github]

Python utilities for stellar model grid interpolation.

If you find this package useful, please cite [Claytor et al. (2020)][gyro paper].

Download the model grids from [Google Drive][google drive].

(C) [Zachary R. Claytor][zclaytor]  
Institute for Astronomy  
University of Hawaiʻi  
2020 April 1

Kīauhōkū  
From Hawaiian:  
1. vt. To sense the span of a star's existence (i.e., its age).  
2. n. The speed of a star (in this case, its rotational speed).  

This name was created in partnership with Dr. Larry Kimura and Bruce Torres Fischer, a student participant in *A Hua He Inoa*, a program to bring Hawaiian naming practices to new astronomical discoveries. We are grateful for their collaboration.

Kīauhōkū is a suite of Python tools to interact with, manipulate, and interpolate between stellar evolutionary tracks in a model grid. In its current version, it is designed to work with the model grid used in [Claytor et al. (2020)][gyro paper], which was generated using YREC with the magnetic braking law of [van Saders et al. (2013)][van Saders]. 

I am currently in the process of adapting the code to work with a wider range of stellar evolution models, so stay tuned!


Kīauhōkū requires the use of Python 3 and uses the following Python packages:  
- numpy  
- scipy  
- pandas  
- matplotlib  
- tqdm
- pyarrow (or some package that supports parquet files)
- [emcee][emcee]  
- [isochrones][isochrones]


## Space Requirements
The raw model files take up 4.1 GB of space. Once individually pickled and condensed to the EEP basis, the whole set of models + pickled interpolator takes up roughly 5.5 GB.

When loaded into memory, the Grid interpolator takes up ~200 MB of RAM.


## Installation
```bash
git clone https://github.com/zclaytor/kiauhoku
cd kiauhoku
pip install .
```


## I don't care about the documentation. Just let me get started!
1. The parent directory contains a set of install scripts for the model grids. Download the model grids from [Google Drive][google drive], then check the first few lines of the install script, including the path to the grids and the names of the eep_params (which are grid dependent).

2. Open an interactive Python session in the directory with the install script. Do the following (this example uses 'rotevol_fastlaunch.py' as the install script):
```python
import kiauhoku as kh
kh.install_grid('rotevol_fastlaunch')
```

3. You're ready to go! You can run `grid = kh.load_interpolator('fastlaunch')` and interpolate away.


## How it works

We start with output evolution tracks from your favorite stellar modeling software. For YREC, these are the \*.out files. Each \*.out file has, for one specific initial metallicity and alpha-abundance, a series of evolution tracks for a range of initial masses. The grid included with `kiauhoku` has eight \*.out files, corresponding to  
[M/H] ~ [-1.0, -0.5, 0.0, 0.5] and  
[alpha/M] ~ [0.0, 0.4].  
Each file contains 171 evolution tracks for 0.30 <= M/Msun <= 2.00 in steps of 0.01\*Msun.

1. First we load the tracks into a pandas MultiIndexed DataFrame and save to a parquet file.

2. Age is not an optimal dimension for comparing consecutive evolution tracks. For this reason we condense each evolution track in the time domain to a series of Equivalent Evolutionary Phases (EEPs) after the method of Dotter (2016). The EEP-based tracks are packaged into a MultiIndexed DataFrame and saved to parquet.

3. We finally load the EEP-based tracks into a `kiauhoku.stargrid.StarGridInterpolator` object. The `StarGridInterpolator` is based on the DataFrameInterpolator (`DFInterpolator`) from Tim Morton's `isochrones` package. It performs linear interpolation between consecutive evolution tracks for an input mass, metallicity, alpha-abundance, and either age or EEP-index. We then pickle the interpolator so it can be accessed quickly and easily.


## Basic Usage

Once you have everything running, try doing this:  
```python
import kiauhoku as kh
grid = kh.load_interpolator('fastlaunch')
star = grid.get_star_eep(1, 0, 0, 330)
```

If it works, you should get something close to the sun. The arguments to get_star_eep are mass (in solar units), metallicity, alpha-abundance, and EEP index. See the documentation for more details.

Kīauhōkū comes with MCMC functionality through `emcee`. See the jupyter notebook `mcmc.ipynb` for an example.

   
[kiauhoku github]: https://github.com/zclaytor/kiauhoku
[zclaytor]: https://zclaytor.github.io
[gyro paper]: https://ui.adsabs.harvard.edu/abs/2020ApJ...888...43C/abstract
[van Saders]: https://ui.adsabs.harvard.edu/abs/2013ApJ...776...67V/abstract
[emcee]: https://emcee.readthedocs.io/en/latest/
[isochrones]: https://isochrones.readthedocs.io/en/latest/
[google drive]: https://drive.google.com/drive/folders/1JLB-IATwHT4XE8qk3y3cXv6ek7Lhk2-H?usp=sharing