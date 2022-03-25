import pytest
from kiauhoku.stargrid import load_eep_grid, load_interpolator, StarGrid

@pytest.fixture
def fastlaunch_eep():
    return load_eep_grid("fastlaunch")

@pytest.fixture
def fastlaunch_interp():
    return load_interpolator("fastlaunch")
    
def test_get_sun(fastlaunch_eep):
    assert isinstance(fastlaunch_eep.loc[1, 0, 0], StarGrid)

def test_interp_star(fastlaunch_interp):
    star = fastlaunch_interp.get_star_eep((0.875, -0.123, 0.123, 285.5))
    assert(star["Log Teff(K)"] == 3.7376106692485824)