name = "kiauhoku"

from .stargrid import StarGrid, StarGridInterpolator
from .stargrid import install_grid, load_interpolator, download
from .stargrid import load_grid, load_full_grid, load_eep_grid
from .calc_HZ import add_HZ, add_HZ_custom

__version__ = "2.0.0"