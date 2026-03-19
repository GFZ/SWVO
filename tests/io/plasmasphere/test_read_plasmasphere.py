# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from swvo.io.plasmasphere import (
    PlasmasphereDensityCube,
    PlasmaspherePredictionReader,
)


def _build_sample_density_dataframe() -> pd.DataFrame:
    rows = []
    times = [datetime(2026, 1, 1, h, 0, tzinfo=timezone.utc) for h in range(24)]
    for ts in times:
        for mlt in [0.0, 6.0, 12.0]:
            for l_value in [1.5, 4, 6.5]:
                rows.append(
                    {
                        "date": ts,
                        "L": l_value,
                        "MLT": mlt,
                        "predicted_densities_ensemble_member_0": float(l_value + mlt),
                        "predicted_densities_ensemble_member_1": float(2.0 * l_value + mlt),
                    }
                )
    return pd.DataFrame(rows)


class TestPlasmaspherePredictionReader:
    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "plasmasphere"

    @pytest.fixture
    def sample_file(self, data_dir: Path) -> Path:
        data_dir.mkdir(parents=True, exist_ok=True)
        sample_date = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        file_name = f"plasmasphere_density_{sample_date.strftime('%Y%m%dT%H00')}.csv"
        file_path = data_dir / file_name
        _build_sample_density_dataframe().to_csv(file_path, index=False)
        return file_path

    def test_initialization_with_env_var(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        with patch.dict(os.environ, {PlasmaspherePredictionReader.ENV_VAR_NAME: str(data_dir)}):
            reader = PlasmaspherePredictionReader()
            assert reader.data_dir == data_dir

    def test_initialization_without_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                PlasmaspherePredictionReader()

    def test_read_existing_file(self, sample_file: Path):
        reader = PlasmaspherePredictionReader(data_dir=sample_file.parent)
        result = reader.read(requested_date=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
        assert result is not None
        assert "t" in result.columns
        assert len(result) == 216  # 24 hours * 3 MLT bins * 3 L bins

    def test_read_missing_file_returns_none(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        reader = PlasmaspherePredictionReader(data_dir=data_dir)
        result = reader.read(requested_date=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
        assert result is None

    def test_build_density_cube_single_column(self, sample_file: Path):
        reader = PlasmaspherePredictionReader(data_dir=sample_file.parent)
        cube = reader.build_density_cube(
            requested_date=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            density_column="predicted_densities_ensemble_member_1",
        )

        assert cube is not None
        assert isinstance(cube, PlasmasphereDensityCube)
        assert cube.density_column == ["predicted_densities_ensemble_member_1"]
        assert len(cube.density_grid) == 1
        assert cube.density_grid[0].shape == (24, 3, 3)

    def test_build_density_cube_multi_column(self, sample_file: Path):
        reader = PlasmaspherePredictionReader(data_dir=sample_file.parent)
        cube = reader.build_density_cube(requested_date=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))

        assert cube is not None
        assert isinstance(cube.density_column, list)
        assert sorted(cube.density_column) == [
            "predicted_densities_ensemble_member_0",
            "predicted_densities_ensemble_member_1",
        ]
        assert len(cube.density_grid) == 2
        assert all(grid.shape == (24, 3, 3) for grid in cube.density_grid)

    def test_get_density_at_time(self, sample_file: Path):
        reader = PlasmaspherePredictionReader(data_dir=sample_file.parent)
        cube = reader.build_density_cube(
            requested_date=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            density_column="predicted_densities_ensemble_member_1",
        )

        assert cube is not None
        density_slice_at_time_t = cube.get_density_at_time(datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc))

        assert isinstance(density_slice_at_time_t, list)
        assert density_slice_at_time_t[0].shape == (3, 3)

    def test_get_density_at_time_missing_raises(self, sample_file: Path):
        reader = PlasmaspherePredictionReader(data_dir=sample_file.parent)
        cube = reader.build_density_cube(
            requested_date=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            density_column="predicted_densities_ensemble_member_1",
        )

        assert cube is not None
        with pytest.raises(IndexError):
            cube.get_density_at_time(datetime(2026, 1, 2, 23, 0, tzinfo=timezone.utc))


def test_density_cube_equality():
    t = np.array([datetime(2026, 1, 1, 0, 0)])
    l = np.array([2.0, 3.0])  # noqa: E741
    mlt = np.array([0.0, 12.0])
    l_grid = np.array([[2.0, 2.0], [3.0, 3.0]])
    mlt_grid = np.array([[0.0, 12.0], [0.0, 12.0]])
    density_grid = [np.ones((1, 2, 2))]

    cube_a = PlasmasphereDensityCube(
        time=t,
        l=l,
        mlt=mlt,
        l_grid=l_grid,
        mlt_grid=mlt_grid,
        density_grid=density_grid,
        density_column=["predicted_densities_ensemble_member_1"],
    )
    cube_b = PlasmasphereDensityCube(
        time=t.copy(),
        l=l.copy(),
        mlt=mlt.copy(),
        l_grid=l_grid.copy(),
        mlt_grid=mlt_grid.copy(),
        density_grid=[density_grid[0].copy()],
        density_column=["predicted_densities_ensemble_member_1"],
    )

    assert cube_a == cube_b


def test_density_cube_not_equal():
    t = np.array([datetime(2026, 1, 1, 0, 0)])
    l = np.array([2.0, 3.0])  # noqa: E741
    mlt = np.array([0.0, 12.0])
    l_grid = np.array([[2.0, 2.0], [3.0, 3.0]])
    mlt_grid = np.array([[0.0, 12.0], [0.0, 12.0]])
    density_grid = [np.ones((1, 2, 2))]

    cube_a = PlasmasphereDensityCube(
        time=t,
        l=l,
        mlt=mlt,
        l_grid=l_grid,
        mlt_grid=mlt_grid,
        density_grid=density_grid,
        density_column=["predicted_densities_ensemble_member_1"],
    )
    cube_b = PlasmasphereDensityCube(
        time=t.copy(),
        l=l.copy(),
        mlt=mlt.copy(),
        l_grid=l_grid.copy(),
        mlt_grid=mlt_grid.copy(),
        density_grid=[density_grid[0].copy()],
        density_column=["predicted_densities_ensemble_member_0"],
    )

    assert cube_a.diff(cube_b)[0] == "density_column mismatch"
    assert cube_a != cube_b


def test_density_cube_not_equal_different_density_grid_length():
    t = np.array([datetime(2026, 1, 1, 0, 0)])
    l = np.array([2.0, 3.0])  # noqa: E741
    mlt = np.array([0.0, 12.0])
    l_grid = np.array([[2.0, 2.0], [3.0, 3.0]])
    mlt_grid = np.array([[0.0, 12.0], [0.0, 12.0]])
    density_grid = [np.ones((1, 2, 2))]

    with pytest.raises(ValueError):
        _ = PlasmasphereDensityCube(
            time=t,
            l=l,
            mlt=mlt,
            l_grid=l_grid,
            mlt_grid=mlt_grid,
            density_grid=density_grid,
            density_column=["predicted_densities_ensemble_member_0", "predicted_densities_ensemble_member_1"],
        )
