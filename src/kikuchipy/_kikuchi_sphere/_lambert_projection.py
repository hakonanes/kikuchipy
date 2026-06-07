#
# Copyright 2019-2026 the kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.
#

"""Functions for projecting to and from the Lambert projection."""

import numba as nb
import numpy as np
import orix.projections as opr
import orix.vector as ove

from kikuchipy._kikuchi_sphere._interpolation import _bilinear_interpolate_nan_skip


@nb.njit(
    "float64[:, :](float64[:], float64[:])",
    cache=True,
    nogil=True,
    fastmath=True,
    parallel=True,
)
def _lambert_to_sphere(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lambert (X, Y) to vector (x, y, z) projection
    :cite:`callahan2013dynamical`.

    Parameters
    ----------
    x, y
        1D arrays of square grid x and y coordinates with 64-bit
        floating point data type.

    Returns
    -------
    cart
        2D array (n, 3) of vectors. The vectors are not normalized, so
        they might not be on the unit sphere.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = x.size
    cart = np.zeros((n, 3), dtype=np.float64)

    for i in nb.prange(n):
        xi = x[i] * np.sqrt(np.pi / 2)
        yi = y[i] * np.sqrt(np.pi / 2)

        xi_abs = abs(xi)
        yi_abs = abs(yi)

        if max([xi_abs, yi_abs]) == 0:
            cart[i] = [0, 0, 1]
        else:
            if xi_abs <= yi_abs:
                q = 2 * yi * np.sqrt(np.pi - yi**2) / np.pi
                qq = xi * np.pi * 0.25 / yi
                cart[i] = [q * np.sin(qq), q * np.cos(qq), 1 - 2 * yi**2 / np.pi]
            else:
                q = 2 * xi * np.sqrt(np.pi - xi**2) / np.pi
                qq = yi * np.pi * 0.25 / xi
                cart[i] = [q * np.cos(qq), q * np.sin(qq), 1 - 2 * xi**2 / np.pi]

    return cart


def _lambert_to_stereo_coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute fractional stereographic pixel coordinates for each
    point on an (n, n) Lambert grid.

    For each output Lambert pixel at grid position (i, j), the
    corresponding input location in the stereographic image is returned
    as a fractional (row, column) index.  These coordinates are
    independent of the pattern data and need only be computed once per
    signal shape.

    Parameters
    ----------
    n
        Side length of the square Lambert output grid.

    Returns
    -------
    row_frac
        1D float64 array of length n*n with fractional row indices into
        the stereographic image, in row-major (C) order.
    col_frac
        1D float64 array of length n*n with fractional column indices
        into the stereographic image, in row-major (C) order.
    """
    arr = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    x_lambert, y_lambert = np.meshgrid(arr, arr)
    xyz = _lambert_to_sphere(x_lambert.ravel(), y_lambert.ravel())
    sp = opr.StereographicProjection()
    x_stereo, y_stereo = sp.vector2xy(ove.Vector3d(xyz))
    # Map from [-1, 1] stereographic coordinates to [0, n-1] pixel indices
    scale = (n - 1) / 2.0
    col_frac = np.ascontiguousarray((x_stereo + 1.0) * scale)
    row_frac = np.ascontiguousarray((y_stereo + 1.0) * scale)
    return row_frac, col_frac


def _project_stereo_to_lambert(
    pattern_stereo: np.ndarray, row_frac: np.ndarray, col_frac: np.ndarray
) -> np.ndarray:
    """Project a single master pattern from the stereographic to the
    Lambert projection using bilinear interpolation.

    Parameters
    ----------
    pattern_stereo
        2D array containing the master pattern in the stereographic
        projection. Intensities outside the equator are ignored.
    row_frac
        Precomputed fractional row indices into *pattern_stereo* for
        each Lambert output pixel, as returned by
        :func:`_lambert_to_stereo_coords`.
    col_frac
        Precomputed fractional column indices into *pattern_stereo* for
        each Lambert output pixel.

    Returns
    -------
    out
        1D float64 array (n*n,) with interpolated intensities in the
        Lambert projection, in row-major order. NaN where the Lambert
        coordinate maps outside the stereographic image.
    """
    img = np.array(pattern_stereo, dtype=np.float64)

    # Set points outside equator to NaN. Verified that all masked out
    # intensities are 0 in master patterns of shape (1001, 1001) from
    # EMsoft.
    n = img.shape[0]
    arr = np.linspace(-1.0, 1.0, n)
    xs, ys = np.meshgrid(arr, arr[::-1])
    img[xs**2 + ys**2 > 1.0] = np.nan

    img = np.ascontiguousarray(img)

    out = _bilinear_interpolate_nan_skip(
        image=img, row_frac=row_frac, col_frac=col_frac
    )

    return out
