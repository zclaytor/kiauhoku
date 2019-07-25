# [Kīauhōkū][kiauhoku github]

Python utilities for stellar model grid interpolation

(C) [Zachary R. Claytor][zclaytor]  
Institute for Astronomy  
University of Hawaiʻi  
2019 July 1  

Kīauhōkū  
From Hawaiian:  
1. vt. To sense the span of a star's existence (i.e., its age).  
2. n. The speed of a star (in this case, its rotational speed).  

This name was created in partnership with Dr. Larry Kimura and Bruce Torres Fischer, a student participant in *A Hua He Inoa*, a program to bring Hawaiian naming practices to new astronomical discoveries. We are grateful for their collaboration.

Kīauhōkū is a suite of Python tools to interact with, manipulate, and interpolate between stellar evolutionary tracks in a model grid. In its current version, comes packaged with the model grid used in Claytor et al. (2019, in prep), which was generated using YREC with the magnetic braking law of van Saders et al. (2013). See literature for details.

Kīauhōkū requires the use of Python 3 and uses the following Python packages:  
- numpy  
- scipy  
- pandas  
- matplotlib  
- [emcee-3][emcee] * See Dan Foreman-Mackey's webpage to get the latest version  
- tqdm  
- multiprocessing  


## Space Requirements
The raw model files take up 4.1 GB of space. Once individually pickled and condensed to the EEP basis, the whole set of models + pickled interpolator takes up roughly 6.5 GB.

When loaded into memory, the Grid interpolator takes up ~250 MB of RAM.


## I don't care about the documentation. Just let me get started!
1. Check the first few lines of the config.py file to make sure the paths point to the directories you want them to.

2. run setup.py (This should take ~10 to 15 minutes to run).

3. You're ready to go! from kiauhoku import load_interpolator and decide which labels you want to use/get, OR check the Jupyter notebook for a guided tutorial and MCMC application.


## How it works

We start with output evolution tracks from your favorite stellar modeling software. For YREC, these are the \*.out files. Each \*.out file has, for one specific initial metallicity and alpha-abundance, a series of evolution tracks for a range of initial masses. The grid included with kiauhoku has eight \*.out files, corresponding to  
[M/H] ~ [-1.0, -0.5, 0.0, 0.5] and  
[alpha/M] ~ [0.0, 0.4].  
Each file contains 171 evolution tracks for 0.30 <= M/Msun <= 2.00 in steps of 0.01\*Msun.

1. First we pickle each evolution track for easier access.

2. Age is not an optimal dimension for comparing consecutive evolution tracks. For this reason we condense each evolution track in the time domain to a series of Equivalent Evolutionary Phases (EEPs) after the method of Dotter (2016). The EEP-based tracks are then pickled.

3. We finally load the EEP-based tracks into a kiauhoku.Grid object. Grid is based on scipy.interpolate.RegularGridInterpolator and performs linear interpolation between consecutive evolution tracks for an input mass, metallicity, alpha-abundance, and either age or EEP-index. We pickle the Grid so it can be accessed quickly and easily.


## Basic Usage

Once you have everything running, try doing this:  
    > `from kiauhoku import load_interpolator`  
    > `grid = load_interpolator(["Age(Gyr)", "Prot(days)", "L/Lsun"])`  
    > `grid.get_star(1, 0, 0, age=4.5)`  

If it works, you should get something close to the sun. The arguments to get_star are mass (in solar units), metallicity, and alpha-abundance. See the documentation in kiauhoku.py for more details.

Kīauhōkū comes with MCMC functionality through emcee-3. See the docs for kiauhoku.Grid.mcmc for more details.


## Contents

- model_grid_utils.py
  Contains methods for loading, interacting with, and pickling the raw models from their source files.

- eep_functions.py
  Contains methods for condensing the raw evolution tracks to EEP basis. EEPs can be defined by the user using methods contained here. Also has some plotting/testing methods.

- config.py
  Contains path variables for the locations of the raw evolution track files, EEP-based track files, and interpolator file. Also contains definitions of model grid boundaries and some indexing variables for other methods to use.

- Grid.py
  Contains the Grid class, which is used to hold the interpolator objects. The Grid object has methods for using MCMC to take samples from the model grid.
  
   
[kiauhoku github]: https://github.com/zclaytor/kiauhoku
[zclaytor]: https://zclaytor.github.io
[emcee]: https://emcee.readthedocs.io/en/latest/
