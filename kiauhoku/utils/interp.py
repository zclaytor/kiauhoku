'''
interp.py
from Tim Morton's isochrones package:
https://github.com/timothydmorton/isochrones

Copyright (c) 2014 timothydmorton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import itertools

import numba as nb
import numpy as np
import pandas as pd


@nb.jit(nopython=True)
def searchsorted(arr, x, N=-1):
    """N is length of arr
    """
    if N == -1:
        N = len(arr)

    L = 0
    R = N - 1
    done = False
    eq = False
    m = (L + R) // 2
    while not done:
        xm = arr[m]
        if xm < x:
            L = m + 1
        elif xm > x:
            R = m - 1
        elif xm == x:
            L = m
            eq = True
            done = True
        m = (L + R) // 2
        if L > R:
            done = True
    return L, eq


@nb.jit(nopython=True)
def find_indices(point, iis):
    ndim = len(point)

    indices = np.zeros(ndim, dtype=nb.uint32)
    norm_distances = np.zeros(ndim, dtype=nb.float64)
    out_of_bounds = False
    for i in range(ndim):
        ii = iis[i]
        n = len(ii)
        x = point[i]
        ix, eq = searchsorted(ii, x)
        if eq:
            indices[i] = ix
            norm_distances[i] = 0
        else:
            ix = ix - 1
            indices[i] = ix
            dx = ii[ix + 1] - ii[ix]
            norm_distances[i] = (x - ii[ix]) / dx
        out_of_bounds &= x < ii[0] or x > ii[n - 1]

    return indices, norm_distances, out_of_bounds


@nb.jit(nopython=True)
def find_indices_2d(x0, x1, ii0, ii1):

    n0 = len(ii0)
    n1 = len(ii1)

    indices = np.empty(2, dtype=nb.uint32)
    norm_distances = np.empty(2, dtype=nb.float64)

    if (x0 < ii0[0]) or (x0 > ii0[n0 - 1]) or (x1 < ii1[0]) or (x1 > ii1[n1 - 1]):
        return indices, norm_distances, True  # Out of bounds

    ix, eq = searchsorted(ii0, x0)
    if eq:
        indices[0] = ix
        norm_distances[0] = 0
    else:
        indices[0] = ix - 1
        c0 = ii0[ix - 1]
        norm_distances[0] = (x0 - c0) / (ii0[ix] - c0)

    ix, eq = searchsorted(ii1, x1)
    if eq:
        indices[1] = ix
        norm_distances[1] = 0
    else:
        indices[1] = ix - 1
        c0 = ii1[ix - 1]
        norm_distances[1] = (x1 - c0) / (ii1[ix] - c0)

    return indices, norm_distances, False


@nb.jit(nopython=True)
def find_indices_3d(x0, x1, x2, ii0, ii1, ii2):

    n0 = len(ii0)
    n1 = len(ii1)
    n2 = len(ii2)

    indices = np.empty(3, dtype=nb.uint32)
    norm_distances = np.empty(3, dtype=nb.float64)

    if (
        (x0 < ii0[0])
        or (x0 > ii0[n0 - 1])
        or (x1 < ii1[0])
        or (x1 > ii1[n1 - 1])
        or (x2 < ii2[0])
        or (x2 > ii2[n2 - 1])
    ):
        return indices, norm_distances, True  # Out of bounds

    ix, eq = searchsorted(ii0, x0)
    if eq:
        indices[0] = ix
        norm_distances[0] = 0
    else:
        indices[0] = ix - 1
        c0 = ii0[ix - 1]
        norm_distances[0] = (x0 - c0) / (ii0[ix] - c0)

    ix, eq = searchsorted(ii1, x1)
    if eq:
        indices[1] = ix
        norm_distances[1] = 0
    else:
        indices[1] = ix - 1
        c0 = ii1[ix - 1]
        norm_distances[1] = (x1 - c0) / (ii1[ix] - c0)

    ix, eq = searchsorted(ii2, x2)
    if eq:
        indices[2] = ix
        norm_distances[2] = 0
    else:
        indices[2] = ix - 1
        c0 = ii2[ix - 1]
        norm_distances[2] = (x2 - c0) / (ii2[ix] - c0)

    return indices, norm_distances, False


@nb.jit(nopython=True)
def find_indices_4d(x0, x1, x2, x3, ii0, ii1, ii2, ii3):

    n0 = len(ii0)
    n1 = len(ii1)
    n2 = len(ii2)
    n3 = len(ii3)

    indices = np.empty(4, dtype=nb.uint32)
    norm_distances = np.empty(4, dtype=nb.float64)

    if (
        (x0 < ii0[0])
        or (x0 > ii0[n0 - 1])
        or (x1 < ii1[0])
        or (x1 > ii1[n1 - 1])
        or (x2 < ii2[0])
        or (x2 > ii2[n2 - 1])
        or (x3 < ii3[0])
        or (x3 > ii3[n3 - 1])
    ):
        return indices, norm_distances, True  # Out of bounds

    ix, eq = searchsorted(ii0, x0)
    if eq:
        indices[0] = ix
        norm_distances[0] = 0
    else:
        indices[0] = ix - 1
        c0 = ii0[ix - 1]
        norm_distances[0] = (x0 - c0) / (ii0[ix] - c0)

    ix, eq = searchsorted(ii1, x1)
    if eq:
        indices[1] = ix
        norm_distances[1] = 0
    else:
        indices[1] = ix - 1
        c0 = ii1[ix - 1]
        norm_distances[1] = (x1 - c0) / (ii1[ix] - c0)

    ix, eq = searchsorted(ii2, x2)
    if eq:
        indices[2] = ix
        norm_distances[2] = 0
    else:
        indices[2] = ix - 1
        c0 = ii2[ix - 1]
        norm_distances[2] = (x2 - c0) / (ii2[ix] - c0)

    ix, eq = searchsorted(ii3, x3)
    if eq:
        indices[3] = ix
        norm_distances[3] = 0
    else:
        indices[3] = ix - 1
        c0 = ii3[ix - 1]
        norm_distances[3] = (x3 - c0) / (ii3[ix] - c0)

    return indices, norm_distances, False


@nb.jit(nopython=True)
def find_indices_5d(x0, x1, x2, x3, x4, ii0, ii1, ii2, ii3, ii4):

    n0 = len(ii0)
    n1 = len(ii1)
    n2 = len(ii2)
    n3 = len(ii3)
    n4 = len(ii4)

    indices = np.empty(5, dtype=nb.uint32)
    norm_distances = np.empty(5, dtype=nb.float64)

    if (
        (x0 < ii0[0])
        or (x0 > ii0[n0 - 1])
        or (x1 < ii1[0])
        or (x1 > ii1[n1 - 1])
        or (x2 < ii2[0])
        or (x2 > ii2[n2 - 1])
        or (x3 < ii3[0])
        or (x3 > ii3[n3 - 1])
        or (x4 < ii4[0])
        or (x4 > ii4[n4 - 1])
    ):
        return indices, norm_distances, True  # Out of bounds

    ix, eq = searchsorted(ii0, x0)
    if eq:
        indices[0] = ix
        norm_distances[0] = 0
    else:
        indices[0] = ix - 1
        c0 = ii0[ix - 1]
        norm_distances[0] = (x0 - c0) / (ii0[ix] - c0)

    ix, eq = searchsorted(ii1, x1)
    if eq:
        indices[1] = ix
        norm_distances[1] = 0
    else:
        indices[1] = ix - 1
        c0 = ii1[ix - 1]
        norm_distances[1] = (x1 - c0) / (ii1[ix] - c0)

    ix, eq = searchsorted(ii2, x2)
    if eq:
        indices[2] = ix
        norm_distances[2] = 0
    else:
        indices[2] = ix - 1
        c0 = ii2[ix - 1]
        norm_distances[2] = (x2 - c0) / (ii2[ix] - c0)

    ix, eq = searchsorted(ii3, x3)
    if eq:
        indices[3] = ix
        norm_distances[3] = 0
    else:
        indices[3] = ix - 1
        c0 = ii3[ix - 1]
        norm_distances[3] = (x3 - c0) / (ii3[ix] - c0)

    ix, eq = searchsorted(ii4, x4)
    if eq:
        indices[4] = ix
        norm_distances[4] = 0
    else:
        indices[4] = ix - 1
        c0 = ii4[ix - 1]
        norm_distances[4] = (x4 - c0) / (ii4[ix] - c0)

    return indices, norm_distances, False


@nb.jit(nopython=True)
def find_indices_6d(x0, x1, x2, x3, x4, x5, ii0, ii1, ii2, ii3, ii4, ii5):

    n0 = len(ii0)
    n1 = len(ii1)
    n2 = len(ii2)
    n3 = len(ii3)
    n4 = len(ii4)
    n5 = len(ii5)

    indices = np.empty(6, dtype=nb.uint32)
    norm_distances = np.empty(6, dtype=nb.float64)

    if (
        (x0 < ii0[0])
        or (x0 > ii0[n0 - 1])
        or (x1 < ii1[0])
        or (x1 > ii1[n1 - 1])
        or (x2 < ii2[0])
        or (x2 > ii2[n2 - 1])
        or (x3 < ii3[0])
        or (x3 > ii3[n3 - 1])
        or (x4 < ii4[0])
        or (x4 > ii4[n4 - 1])
        or (x5 < ii5[0])
        or (x5 > ii5[n5 - 1])
    ):
        return indices, norm_distances, True  # Out of bounds

    ix, eq = searchsorted(ii0, x0)
    if eq:
        indices[0] = ix
        norm_distances[0] = 0
    else:
        indices[0] = ix - 1
        c0 = ii0[ix - 1]
        norm_distances[0] = (x0 - c0) / (ii0[ix] - c0)

    ix, eq = searchsorted(ii1, x1)
    if eq:
        indices[1] = ix
        norm_distances[1] = 0
    else:
        indices[1] = ix - 1
        c0 = ii1[ix - 1]
        norm_distances[1] = (x1 - c0) / (ii1[ix] - c0)

    ix, eq = searchsorted(ii2, x2)
    if eq:
        indices[2] = ix
        norm_distances[2] = 0
    else:
        indices[2] = ix - 1
        c0 = ii2[ix - 1]
        norm_distances[2] = (x2 - c0) / (ii2[ix] - c0)

    ix, eq = searchsorted(ii3, x3)
    if eq:
        indices[3] = ix
        norm_distances[3] = 0
    else:
        indices[3] = ix - 1
        c0 = ii3[ix - 1]
        norm_distances[3] = (x3 - c0) / (ii3[ix] - c0)

    ix, eq = searchsorted(ii4, x4)
    if eq:
        indices[4] = ix
        norm_distances[4] = 0
    else:
        indices[4] = ix - 1
        c0 = ii4[ix - 1]
        norm_distances[4] = (x4 - c0) / (ii4[ix] - c0)

    ix, eq = searchsorted(ii5, x5)
    if eq:
        indices[5] = ix
        norm_distances[5] = 0
    else:
        indices[5] = ix - 1
        c0 = ii5[ix - 1]
        norm_distances[5] = (x5 - c0) / (ii5[ix] - c0)

    return indices, norm_distances, False


@nb.jit(nopython=True)
def interp_value_2d(x0, x1, grid, icols, ii0, ii1):
    if x0 != x0 or x1 != x1:
        return np.array([np.nan for i in icols])

    indices, norm_distances, out_of_bounds = find_indices_2d(x0, x1, ii0, ii1)

    if out_of_bounds:
        return np.array([np.nan for i in icols])
    # The following should be equivalent to
    #  edges = np.array(list(itertools.product(*[[i, i+1] for i in indices])))

    ndim = 2
    n_edges = 2 ** ndim
    edges = np.zeros((n_edges, ndim))
    for i in range(n_edges):
        for j in range(ndim):
            edges[i, j] = indices[j] + ((i >> (ndim - 1 - j)) & 1)  # woohoo!

    n_values = len(icols)
    values = np.zeros(n_values, dtype=nb.float64)

    for j in range(n_edges):
        edge_indices = np.zeros(ndim, dtype=nb.uint32)
        for k in range(ndim):
            edge_indices[k] = edges[j, k]

        weight = 1.0
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            if ei == i:
                weight *= 1 - yi
            else:
                weight *= yi

        for i_icol in range(n_values):
            icol = icols[i_icol]

            # Now, get the value; this is why general ND doesn't work
            grid_indices = (edge_indices[0], edge_indices[1], icol)
            values[i_icol] += grid[grid_indices] * weight

    return values


@nb.jit(nopython=True)
def interp_value_3d(x0, x1, x2, grid, icols, ii0, ii1, ii2):
    if x0 != x0 or x1 != x1 or x2 != x2:
        return np.array([np.nan for i in icols])

    indices, norm_distances, out_of_bounds = find_indices_3d(x0, x1, x2, ii0, ii1, ii2)

    if out_of_bounds:
        return np.array([np.nan for i in icols])
    # The following should be equivalent to
    #  edges = np.array(list(itertools.product(*[[i, i+1] for i in indices])))

    ndim = 3
    n_edges = 2 ** ndim
    edges = np.zeros((n_edges, ndim))
    for i in range(n_edges):
        for j in range(ndim):
            edges[i, j] = indices[j] + ((i >> (ndim - 1 - j)) & 1)  # woohoo!

    n_values = len(icols)
    values = np.zeros(n_values, dtype=nb.float64)

    for j in range(n_edges):
        edge_indices = np.zeros(ndim, dtype=nb.uint32)
        for k in range(ndim):
            edge_indices[k] = edges[j, k]

        weight = 1.0
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            if ei == i:
                weight *= 1 - yi
            else:
                weight *= yi

        for i_icol in range(n_values):
            icol = icols[i_icol]

            # Now, get the value; this is why general ND doesn't work
            grid_indices = (edge_indices[0], edge_indices[1], edge_indices[2], icol)
            values[i_icol] += grid[grid_indices] * weight

    return values


@nb.jit(nopython=True)
def interp_value_4d(x0, x1, x2, x3, grid, icols, ii0, ii1, ii2, ii3):
    if x0 != x0 or x1 != x1 or x2 != x2 or x3 != x3:
        return np.array([np.nan for i in icols])

    indices, norm_distances, out_of_bounds = find_indices_4d(x0, x1, x2, x3, ii0, ii1, ii2, ii3)

    if out_of_bounds:
        return np.array([np.nan for i in icols])

    # The following should be equivalent to
    #  edges = np.array(list(itertools.product(*[[i, i+1] for i in indices])))

    ndim = 4
    n_edges = 2 ** ndim
    edges = np.zeros((n_edges, ndim))
    for i in range(n_edges):
        for j in range(ndim):
            edges[i, j] = indices[j] + ((i >> (ndim - 1 - j)) & 1)  # woohoo!

    n_values = len(icols)
    values = np.zeros(n_values, dtype=nb.float64)

    for j in range(n_edges):
        edge_indices = np.zeros(ndim, dtype=nb.uint32)
        for k in range(ndim):
            edge_indices[k] = edges[j, k]

        weight = 1.0
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            if ei == i:
                weight *= 1 - yi
            else:
                weight *= yi

        for i_icol in range(n_values):
            icol = icols[i_icol]

            # Now, get the value; this is why general ND doesn't work
            grid_indices = (edge_indices[0], edge_indices[1], edge_indices[2], edge_indices[3], icol)
            values[i_icol] += grid[grid_indices] * weight

    return values


@nb.jit(nopython=True)
def interp_value_5d(x0, x1, x2, x3, x4, grid, icols, ii0, ii1, ii2, ii3, ii4):
    if x0 != x0 or x1 != x1 or x2 != x2 or x3 != x3 or x4 != x4:
        return np.array([np.nan for i in icols])

    indices, norm_distances, out_of_bounds = find_indices_5d(x0, x1, x2, x3, x4, ii0, ii1, ii2, ii3, ii4)

    if out_of_bounds:
        return np.array([np.nan for i in icols])

    # The following should be equivalent to
    #  edges = np.array(list(itertools.product(*[[i, i+1] for i in indices])))

    ndim = 5
    n_edges = 2 ** ndim
    edges = np.zeros((n_edges, ndim))
    for i in range(n_edges):
        for j in range(ndim):
            edges[i, j] = indices[j] + ((i >> (ndim - 1 - j)) & 1)  # woohoo!

    n_values = len(icols)
    values = np.zeros(n_values, dtype=nb.float64)

    for j in range(n_edges):
        edge_indices = np.zeros(ndim, dtype=nb.uint32)
        for k in range(ndim):
            edge_indices[k] = edges[j, k]

        weight = 1.0
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            if ei == i:
                weight *= 1 - yi
            else:
                weight *= yi

        for i_icol in range(n_values):
            icol = icols[i_icol]

            # Now, get the value; this is why general ND doesn't work
            grid_indices = (edge_indices[0], edge_indices[1], edge_indices[2], edge_indices[3], edge_indices[4], icol)
            values[i_icol] += grid[grid_indices] * weight

    return values


@nb.jit(nopython=True)
def interp_value_6d(x0, x1, x2, x3, x4, x5, grid, icols, ii0, ii1, ii2, ii3, ii4, ii5):
    if x0 != x0 or x1 != x1 or x2 != x2 or x3 != x3 or x4 != x4 or x5 != x5:
        return np.array([np.nan for i in icols])

    indices, norm_distances, out_of_bounds = find_indices_6d(x0, x1, x2, x3, x4, x5, ii0, ii1, ii2, ii3, ii4, ii5)

    if out_of_bounds:
        return np.array([np.nan for i in icols])

    # The following should be equivalent to
    #  edges = np.array(list(itertools.product(*[[i, i+1] for i in indices])))

    ndim = 6
    n_edges = 2 ** ndim
    edges = np.zeros((n_edges, ndim))
    for i in range(n_edges):
        for j in range(ndim):
            edges[i, j] = indices[j] + ((i >> (ndim - 1 - j)) & 1)  # woohoo!

    n_values = len(icols)
    values = np.zeros(n_values, dtype=nb.float64)

    for j in range(n_edges):
        edge_indices = np.zeros(ndim, dtype=nb.uint32)
        for k in range(ndim):
            edge_indices[k] = edges[j, k]

        weight = 1.0
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            if ei == i:
                weight *= 1 - yi
            else:
                weight *= yi

        for i_icol in range(n_values):
            icol = icols[i_icol]

            # Now, get the value; this is why general ND doesn't work
            grid_indices = (edge_indices[0], edge_indices[1], edge_indices[2], edge_indices[3], edge_indices[4], edge_indices[5], icol)
            values[i_icol] += grid[grid_indices] * weight

    return values


@nb.jit(nopython=True)
def interp_values_2d(xx0, xx1, grid, icols, ii0, ii1):
    """xx1, xx2, xx3 are all arrays at which values are desired
    """

    N = len(xx0)
    ncols = len(icols)
    results = np.empty((N, ncols), dtype=nb.float64)
    for i in range(N):
        res = interp_value_2d(xx0[i], xx1[i], grid, icols, ii0, ii1)
        for j in range(ncols):
            results[i, j] = res[j]

    return results


@nb.jit(nopython=True)
def interp_values_3d(xx0, xx1, xx2, grid, icols, ii0, ii1, ii2):
    """xx1, xx2, xx3 are all arrays at which values are desired
    """

    N = len(xx0)
    ncols = len(icols)
    results = np.empty((N, ncols), dtype=nb.float64)
    for i in range(N):
        res = interp_value_3d(xx0[i], xx1[i], xx2[i], grid, icols, ii0, ii1, ii2)
        for j in range(ncols):
            results[i, j] = res[j]

    return results


@nb.jit(nopython=True)
def interp_values_4d(xx0, xx1, xx2, xx3, grid, icols, ii0, ii1, ii2, ii3):
    """xx1, xx2, xx3 are all arrays at which values are desired
    """

    N = len(xx0)
    ncols = len(icols)
    results = np.empty((N, ncols), dtype=nb.float64)
    for i in range(N):
        res = interp_value_4d(xx0[i], xx1[i], xx2[i], xx3[i], grid, icols, ii0, ii1, ii2, ii3)
        for j in range(ncols):
            results[i, j] = res[j]

    return results


@nb.jit(nopython=True)
def interp_values_5d(xx0, xx1, xx2, xx3, xx4, grid, icols, ii0, ii1, ii2, ii3, ii4):
    """xx1, xx2, xx3, xx4 are all arrays at which values are desired
    """

    N = len(xx0)
    ncols = len(icols)
    results = np.empty((N, ncols), dtype=nb.float64)
    for i in range(N):
        res = interp_value_5d(xx0[i], xx1[i], xx2[i], xx3[i], xx4[i], grid, icols, ii0, ii1, ii2, ii3, ii4)
        for j in range(ncols):
            results[i, j] = res[j]

    return results


@nb.jit(nopython=True)
def interp_values_6d(xx0, xx1, xx2, xx3, xx4, xx5, grid, icols, ii0, ii1, ii2, ii3, ii4, ii5):
    """xx1, xx2, xx3, xx4, xx5 are all arrays at which values are desired
    """

    N = len(xx0)
    ncols = len(icols)
    results = np.empty((N, ncols), dtype=nb.float64)
    for i in range(N):
        res = interp_value_6d(xx0[i], xx1[i], xx2[i], xx3[i], xx4[i], xx5[i], grid, icols, ii0, ii1, ii2, ii3, ii4, ii5)
        for j in range(ncols):
            results[i, j] = res[j]

    return results


@nb.jit(nopython=True)
def sign(x):
    if x < 0:
        return -1
    else:
        return 1


# @jit(nopython=True)
def find_closest3(
    val, a, b, v1, v2, grid, icol, ii1, ii2, ii3, bisect_tol=0.5, newton_tol=0.01, max_iter=100, debug=False
):
    """Find value of 3rd index array where interp_value is closest to val
    val : value to match
    a, b: min and max x-value to serch
=   x1, x2 : first and second values to pass to inter_values
    grid : 4d grid
    icol : index of value dimension of grid
    ii1, ii2, ii3 : grid dimension arrays
    """

    # First, do a bisect search to get it close
    done = False
    ya = interp_value_3d(v1, v2, a, grid, icol, ii1, ii2, ii3) - val
    yb = interp_value_3d(v1, v2, b, grid, icol, ii1, ii2, ii3) - val
    if debug:
        print("Initial values: {}: {}".format((a, b), (ya, yb)))

    if yb != yb or ya != ya:
        # bounds are nan, return nan.
        return np.nan
    elif abs(ya) < newton_tol:
        return float(a)
    elif abs(yb) < newton_tol:
        return float(b)
    elif ya > 0 and yb > 0:
        return np.nan
    elif yb < 0 and yb < 0:
        return np.nan

    else:
        if debug:
            print("doing bisect search...")

        while not done:
            c = (a + b) / 2
            yc = interp_value_3d(v1, v2, c, grid, icol, ii1, ii2, ii3) - val
            if yc == 0 or (b - a) / 2 < bisect_tol:
                done = True
            if sign(yc) == sign(ya):  # (yc >= 0 and ya >= 0) or (yc < 0 and ya < 0):
                a = c
                ya = yc
            else:
                b = c
                yb = yc
            if debug:
                print("{0} {1}".format((a, b, c), (ya, yb, yc)))

    # Now, use the value at index c to seed Newton-secant algorithm
    tol = 1000.0
    i = 0
    x0 = c
    y0 = yc
    x1 = x0 + 0.1
    y1 = interp_value_3d(v1, v2, x1, grid, icol, ii1, ii2, ii3) - val

    if debug:
        print("Newton-secant method...")
    while tol > newton_tol and i < max_iter:
        newx = (x0 * y1 - x1 * y0) / (y1 - y0)
        x0 = x1
        y0 = y1
        x1 = newx
        y1 = interp_value_3d(v1, v2, x1, grid, icol, ii1, ii2, ii3) - val

        # Boo!
        while not y1 == y1:
            if debug:
                print("{0} {1}".format(x1, y1))
            raise RuntimeError("ran into nan." + "run {} with debug=True to see why.".format((val, v2, v1)))

        if y1 >= 0:
            tol = y1
        else:
            tol = -y1
        i += 1
        if debug:
            print("{0} {1}".format(x1, y1))

    return x1


class DFInterpolator(object):
    """Interpolate column values of DataFrame with full-grid hierarchical index
    """

    def __init__(self, df, filename=None, recalc=False, is_full=False):
        self.filename = filename
        self.is_full = is_full
        self.columns = list(df.columns)
        self.n_columns = len(self.columns)
        self.grid = self._make_grid(df, recalc=recalc)
        self.index_columns = tuple(np.array(l, dtype=float) for l in df.index.levels)
        self.index_names = df.index.names

        self.ndim = len(self.index_columns)

        self.column_index = {c: self.columns.index(c) for c in self.columns}

    def _make_grid(self, df, recalc=False):
        if self.filename is not None and os.path.exists(self.filename) and not recalc:
            d = np.load(self.filename)
            grid = d["grid"]
            columns = d["columns"]
            if not all(columns == self.columns):
                raise ValueError("DataFrame columns do not match columns loaded from full grid!")
        else:
            if not self.is_full:  # Need to make a full grid and pad with nans
                idx = pd.MultiIndex.from_tuples([ixs for ixs in itertools.product(*df.index.levels)])

                # Make an empty dataframe with the completely gridded index, and fill
                grid_df = pd.DataFrame(index=idx, columns=df.columns)
                grid_df.loc[df.index] = df
            else:
                grid_df = df

            shape = [len(l) for l in df.index.levels] + [len(df.columns)]

            grid = np.array(grid_df.values, dtype=float).reshape(shape)

            if self.filename is not None:
                np.savez(self.filename, grid=grid, columns=self.columns)

        return grid

    def add_column(self, values, name):
        newgrid = np.empty((self.grid.shape[:-1]) + (self.n_columns + 1,))
        newgrid[..., :-1] = self.grid
        newgrid[..., -1] = values
        self.column_index[name] = self.n_columns
        self.n_columns += 1
        self.columns += [name]
        self.grid = newgrid

    def find_closest(self, val, lo, hi, v1, v2, col="initial_mass", debug=False):
        icol = self.column_index[col]

        if self.ndim == 3:
            return find_closest3(val, lo, hi, v1, v2, self.grid, icol, *self.index_columns, debug=debug)

    def __call__(self, p, cols="all"):
        if cols == "all":
            icols = np.arange(self.n_columns)
        else:
            icols = np.array([self.column_index[col] for col in cols])

        if self.ndim == 2:
            if all([isinstance(p_i, (int, float)) for p_i in p]):
                args = (*p, self.grid, icols, *self.index_columns)
                values = interp_value_2d(*args)
            else:
                b = np.broadcast(*p)
                pp = [np.atleast_1d(np.resize(x, b.shape)).astype(float) for x in p]
                values = interp_values_2d(*pp, self.grid, icols, *self.index_columns)
        if self.ndim == 3:
            if all([isinstance(p_i, (int, float)) for p_i in p]):
                args = (*p, self.grid, icols, *self.index_columns)
                values = interp_value_3d(*args)
            else:
                b = np.broadcast(*p)
                pp = [np.atleast_1d(np.resize(x, b.shape)).astype(float) for x in p]
                values = interp_values_3d(*pp, self.grid, icols, *self.index_columns)
        elif self.ndim == 4:
            if all([isinstance(p_i, (int, float)) for p_i in p]):
                args = (*p, self.grid, icols, *self.index_columns)
                values = interp_value_4d(*args)
            else:
                b = np.broadcast(*p)
                pp = [np.atleast_1d(np.resize(x, b.shape)).astype(float) for x in p]
                values = interp_values_4d(*pp, self.grid, icols, *self.index_columns)
        elif self.ndim == 5:
            if all([isinstance(p_i, (int, float)) for p_i in p]):
                args = (*p, self.grid, icols, *self.index_columns)
                values = interp_value_5d(*args)
            else:
                b = np.broadcast(*p)
                pp = [np.atleast_1d(np.resize(x, b.shape)).astype(float) for x in p]
                values = interp_values_5d(*pp, self.grid, icols, *self.index_columns)
        elif self.ndim == 6:
            if all([isinstance(p_i, (int, float)) for p_i in p]):
                args = (*p, self.grid, icols, *self.index_columns)
                values = interp_value_6d(*args)
            else:
                b = np.broadcast(*p)
                pp = [np.atleast_1d(np.resize(x, b.shape)).astype(float) for x in p]
                values = interp_values_6d(*pp, self.grid, icols, *self.index_columns)

        return values