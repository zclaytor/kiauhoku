import pytest
from pandas import concat
from kiauhoku.stargrid import load_eep_grid, load_interpolator, StarGrid


@pytest.fixture
def fastlaunch_eep():
    return load_eep_grid("fastlaunch")


@pytest.fixture
def fastlaunch_interp():
    return load_interpolator("fastlaunch")


def test_loc(fastlaunch_eep):
    assert isinstance(fastlaunch_eep.loc[1, 0, 0], StarGrid)


def test_interp_2d(fastlaunch_eep):
    interp = fastlaunch_eep.query("initial_alpha == 0 and initial_met == 0").droplevel(["initial_met", "initial_alpha"]).to_interpolator()
    star = interp.get_star_eep((0.875, 285.5))
    assert(star["Log Teff(K)"] == pytest.approx(3.7267098126685316))
    stars = interp.get_star_eep(((0.955, 1.212), (195.5, 467.2)))
    assert(stars.shape == (2, 30))


def test_interp_3d(fastlaunch_eep):
    interp = fastlaunch_eep.query("initial_alpha == 0").droplevel("initial_alpha").to_interpolator()
    star = interp.get_star_eep((0.875, -0.123, 285.5))
    assert(star["Log Teff(K)"] == pytest.approx(3.738440057554258))
    stars = interp.get_star_eep(((0.955, 1.212), (-0.2, 0.4), (195.5, 467.2)))
    assert(stars.shape == (2, 30))


def test_interp_4d(fastlaunch_eep):
    interp = fastlaunch_eep.to_interpolator()
    star = interp.get_star_eep((0.875, -0.123, 0.123, 285.5))
    assert(star["Log Teff(K)"] == pytest.approx(3.7376106692485824))
    stars = interp.get_star_eep(((0.955, 1.212), (-0.2, 0.4), (0.05, 0.3), (195.5, 467.2)))
    assert(stars.shape == (2, 30))


def test_interp_5d(fastlaunch_eep):
    g = fastlaunch_eep
    g0 = g.reset_index().eval("dummy = 0")
    g1 = g0.copy().eval("dummy = 1")

    g0 = g0.set_index(["initial_mass", "initial_met", "initial_alpha", "dummy", "eep"])
    g1 = g1.set_index(["initial_mass", "initial_met", "initial_alpha", "dummy", "eep"])

    gprime = concat([g0, g1]).to_interpolator()
    star = gprime.get_star_eep(((0.875, -0.123, 0.123, 0.5, 285.5)))
    assert(star["Log Teff(K)"] == pytest.approx(3.7376106692485824))


def test_interp_6d(fastlaunch_eep):
    g = fastlaunch_eep
    g00 = g.reset_index().eval("dummy0 = 0").eval("dummy1 = 0")
    g01 = g00.copy().eval("dummy1 = 1")
    g10 = g00.copy().eval("dummy0 = 1")
    g11 = g10.copy().eval("dummy1 = 1")

    g00 = g00.set_index(["initial_mass", "initial_met", "initial_alpha", "dummy0", "dummy1", "eep"])
    g01 = g01.set_index(["initial_mass", "initial_met", "initial_alpha", "dummy0", "dummy1", "eep"])
    g10 = g10.set_index(["initial_mass", "initial_met", "initial_alpha", "dummy0", "dummy1", "eep"])
    g11 = g11.set_index(["initial_mass", "initial_met", "initial_alpha", "dummy0", "dummy1", "eep"])

    gprime = concat([g00, g01, g10, g11]).to_interpolator()
    star = gprime.get_star_eep(((0.875, -0.123, 0.123, 0.5, 0.5, 285.5)))
    assert(star["Log Teff(K)"] == pytest.approx(3.7376106692485824))    
