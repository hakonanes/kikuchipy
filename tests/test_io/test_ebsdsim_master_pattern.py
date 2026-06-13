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

"""Tests for the ebsdsim master pattern reader."""

import pytest

_ = pytest.importorskip("ebsdsim", reason="ebsdsim is not installed")
import numpy as np

import kikuchipy as kp
from kikuchipy.io.plugins.ebsdsim_master_pattern import file_reader


@pytest.mark.gpu
class TestEbsdsimMasterPatternReader:
    # The Ni file has halfw=10 (side=21), 2 energy bins ([15, 5] keV),
    # and hemisphere_dim=1 (centrosymmetric → only northern hemisphere stored).

    def test_default_load(self, ebsdsim_master_pattern_file):
        mp = kp.load(ebsdsim_master_pattern_file)

        assert isinstance(mp, kp.signals.EBSDMasterPattern)
        assert mp.data.shape == (2, 21, 21)
        assert mp.projection == "lambert"
        assert mp.hemisphere == "upper"

    def test_phase(self, ebsdsim_master_pattern_file):
        mp: kp.signals.EBSDMasterPattern = kp.load(ebsdsim_master_pattern_file)

        assert mp.phase.name == "Ni"
        assert mp.phase.space_group.number == 225  # Fm-3m
        assert str(mp.phase.point_group.name) == "m-3m"

    def test_metadata(self, ebsdsim_master_pattern_file):
        mp = kp.load(ebsdsim_master_pattern_file)

        assert mp.metadata["Signal"]["signal_type"] == "EBSDMasterPattern"
        assert mp.metadata["General"]["title"] == "ni_master_pattern"

    def test_energy_axis(self, ebsdsim_master_pattern_file):
        mp = kp.load(ebsdsim_master_pattern_file)

        energy_axis = mp.axes_manager["energy"]
        assert energy_axis.size == 2
        assert energy_axis.units == "keV"
        # Bins are at 15 and 5 keV; offset is the first bin, scale is the step.
        np.testing.assert_allclose(energy_axis.axis, [15.0, 5.0], atol=1.0)

    def test_signal_axes(self, ebsdsim_master_pattern_file):
        mp = kp.load(ebsdsim_master_pattern_file)

        assert mp.axes_manager["height"].size == 21
        assert mp.axes_manager["width"].size == 21
        assert mp.axes_manager["height"].units == "px"
        assert mp.axes_manager["width"].units == "px"

    def test_hemisphere_both(self, ebsdsim_master_pattern_file):
        mp = kp.load(ebsdsim_master_pattern_file, hemisphere="both")

        # Centrosymmetric: northern hemisphere is duplicated to form the south.
        assert mp.data.shape == (2, 2, 21, 21)
        assert mp.hemisphere == "both"
        np.testing.assert_array_equal(mp.data[0], mp.data[1])

    def test_hemisphere_lower_raises_for_centrosymmetric(
        self, ebsdsim_master_pattern_file
    ):
        with pytest.raises(ValueError, match="lower hemisphere"):
            kp.load(ebsdsim_master_pattern_file, hemisphere="lower")

    def test_invalid_hemisphere_raises(self, ebsdsim_master_pattern_file):
        with pytest.raises(ValueError, match="Hemisphere must be one of"):
            kp.load(ebsdsim_master_pattern_file, hemisphere="north")

    def test_file_reader_returns_list(self, ebsdsim_master_pattern_file):
        result = file_reader(ebsdsim_master_pattern_file)

        assert isinstance(result, list)
        assert len(result) == 1
        d = result[0]
        assert set(d).issuperset(
            {
                "data",
                "axes",
                "metadata",
                "original_metadata",
                "phase",
                "projection",
                "hemisphere",
            }
        )
        assert d["projection"] == "lambert"
        assert isinstance(d["phase"], dict)
