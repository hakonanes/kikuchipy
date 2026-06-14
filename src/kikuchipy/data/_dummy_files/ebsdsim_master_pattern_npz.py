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

"""Creation of a small EBSD master pattern simulation using ebsdsim for
testing and IO documentation.
"""

from pathlib import Path

import ebsdsim as es


def create_small_ebsdsim_npz_file(path: Path) -> None:
    mp = es.master_pattern_from_cif(
        "Ni.cif",
        halfw=10,
        voltage_kv=20.0,
        sigma_deg=70.0,
        omega_deg=0.0,
        energy_binwidth_keV=10.0,
        mc_backend="surrogate",
        rank=3,
        dmin=0.2,
    )
    es.save_master_pattern(mp, path)
