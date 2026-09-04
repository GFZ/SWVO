# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swvo.io.exceptions import ModelError
from swvo.io.solar_wind import (
    AVERAGE_VALUES_TO_FILL,
    DSCOVR,
    SWACE,
    SWENLIL,
    SWENLIL_BKG,
    SWENLIL_CME,
    SWIMAP,
    SWMIDL,
    SWOMNI,
    SWSWIFTEnsemble,
    read_solar_wind_from_multiple_models,
)
from swvo.io.solar_wind.read_solar_wind_from_multiple_models import (
    _interpolate_short_gaps,
)

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))
READ_SW_MODULE = importlib.import_module("swvo.io.solar_wind.read_solar_wind_from_multiple_models")

ENLIL_COLUMNS = ["bx_gsm", "by_gsm", "bz_gsm", "bavg", "speed", "proton_density", "temperature", "pdyn"]


def _write_enlil_run(
    model: SWENLIL,
    mode: str,
    date: datetime,
    run_time: str,
    value: float,
    days: int = 12,
    cadence: timedelta = timedelta(seconds=134.3),
) -> None:
    """Write one processed ENLIL run on the model's own irregular cadence."""
    path = model.data_dir / date.strftime("%Y/%m") / f"ENLIL_FORECAST_{mode}_{date:%Y%m%d}_{run_time}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = int(timedelta(days=days) / cadence)
    index = pd.DatetimeIndex([date + cadence * i for i in range(steps)], name="t")
    df = pd.DataFrame({col: np.full(len(index), value) for col in ENLIL_COLUMNS}, index=index)
    df["file_name"] = str(path)
    df.to_csv(path)


class TestReadSolarWindFromMultipleModels:
    @pytest.fixture(scope="session", autouse=True)
    def set_env_var(self, tmp_path_factory):
        # ENLIL runs are generated rather than committed: a realistic run is a few thousand rows
        # and the shared data directory is already tens of megabytes.
        enlil_dir = tmp_path_factory.mktemp("enlil_stream")
        for run_date in (datetime(2024, 11, 22, tzinfo=timezone.utc), datetime(2024, 11, 25, tzinfo=timezone.utc)):
            _write_enlil_run(SWENLIL_BKG(enlil_dir), "bkg", run_date, "0000", 300.0, cadence=timedelta(hours=1))
            _write_enlil_run(SWENLIL_CME(enlil_dir), "cme", run_date, "1800", 403.0, cadence=timedelta(hours=1))

        ENV_VAR_NAMES = {
            "OMNI_HIGH_RES_STREAM_DIR": f"{str(DATA_DIR)}/OMNI",
            "SWIFT_ENSEMBLE_OUTPUT_DIR": f"{str(DATA_DIR)}/ensemble",
            "RT_SW_ACE_STREAM_DIR": f"{str(DATA_DIR)}/ACE_RT",
            "SW_DSCOVR_STREAM_DIR": f"{str(DATA_DIR)}/DSCOVR",
            "SW_ENLIL_STREAM_DIR": str(enlil_dir),
        }

        for key, var in ENV_VAR_NAMES.items():
            os.environ[key] = ENV_VAR_NAMES[key]

    @pytest.fixture
    def sample_times(self):
        now = datetime(2024, 11, 25).replace(tzinfo=timezone.utc, minute=0, second=0, microsecond=0)
        return {
            "past_start": now - timedelta(days=5),
            "past_end": now - timedelta(days=2),
            "future_start": now + timedelta(days=1),
            "future_end": now + timedelta(days=3),
            # far enough out to outrun the SWIFT ensemble, which reaches 2024-12-02
            "far_future_end": now + timedelta(days=9),
            "test_time_now": now,
        }

    @pytest.fixture
    def expected_columns(self):
        return [
            "proton_density",
            "speed",
            "bavg",
            "temperature",
            "bx_gsm",
            "by_gsm",
            "bz_gsm",
            "file_name",
        ]

    @pytest.fixture
    def enlil_instance(self, tmp_path):
        return SWENLIL_CME(tmp_path / "enlil")

    @pytest.fixture
    def enlil_bkg_instance(self, tmp_path):
        return SWENLIL_BKG(tmp_path / "enlil")

    def test_basic_historical_read(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["past_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWENLIL_CME()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in expected_columns)
        assert data.loc["2024-11-20 13:23:00+00:00"].model == "dscovr"
        assert data.loc["2024-11-22 18:00:00+00:00"].model == "omni"
        # no ace since dscovr and ace files are same in the test data and dscovr is before ace in the model_order
        assert not data["file_name"].isna().all()

    def test_basic_forecast_read(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            model_order=[SWSWIFTEnsemble()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )
        assert all(isinstance(d, pd.DataFrame) for d in data)
        assert all(not d["file_name"].isna().all() for d in data)

        assert all(all(d.model == "swift") for d in data)
        for d in data:
            assert all(col in d.columns for col in expected_columns)

    def test_full_ensemble(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            model_order=[SWSWIFTEnsemble()],
            reduce_ensemble=None,
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, list)
        assert len(data) > 1
        assert all(isinstance(d, pd.DataFrame) for d in data)
        assert all(not d["file_name"].isna().all() for d in data)
        for d in data:
            assert all(col in d.columns for col in expected_columns)

    def test_time_ordering_and_transition(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble(), SWENLIL_CME()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        for d in data:
            assert d.index.is_monotonic_increasing
            assert d.loc["2024-11-20 00:00:00+00:00"].model == "omni"
            assert d.loc["2024-11-25 00:01:00+00:00"].model == "swift"
            assert all(col in d.columns for col in expected_columns)

    def test_full_chain_hands_over_from_swift_to_enlil(self, sample_times, expected_columns):
        """Over a window longer than SWIFT reaches, the three tiers run in order: historical up to
        the cutoff, then the SWIFT ensemble, then ENLIL for whatever SWIFT does not cover."""
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["far_future_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble(), SWENLIL_CME()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert len(data) > 1
        for d in data:
            assert d.index.is_monotonic_increasing
            assert all(col in d.columns for col in expected_columns)

            historical = d.index[d["model"].isin(["omni", "dscovr", "ace"])]
            swift = d.index[d["model"] == "swift"]
            enlil = d.index[d["model"] == "enlil_cme"]

            assert len(historical) > 0 and len(swift) > 0 and len(enlil) > 0
            # no historical data past "now", and the two forecasts never overlap
            assert historical.max() <= sample_times["test_time_now"]
            assert swift.min() > sample_times["test_time_now"]
            assert enlil.min() > swift.max()

    def test_forecast_in_past(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["past_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble(), SWENLIL_CME()],
            historical_data_cutoff_time=sample_times["test_time_now"] - timedelta(days=3),
        )

        for d in data:
            assert d.index.is_monotonic_increasing
            assert d.loc["2024-11-22 00:01:00+00:00"].model == "swift"
            assert d.loc["2024-11-22 00:00:00+00:00"].model == "omni"
            assert all(col in d.columns for col in expected_columns)

    def test_time_boundaries(self, sample_times):
        start = sample_times["past_start"]
        end = sample_times["future_end"]

        data = read_solar_wind_from_multiple_models(
            start_time=start,
            end_time=end,
            model_order=[SWOMNI(), SWSWIFTEnsemble(), SWENLIL_CME()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        for d in data:
            assert d.index.min() >= start
            assert d.index.max() <= end

    @pytest.mark.parametrize("offset", [timedelta(0), timedelta(seconds=30), timedelta(microseconds=500)])
    def test_start_time_is_floored_to_the_minute(self, sample_times, offset):
        """The output lives on a one minute grid phased by start_time. Seconds on start_time used
        to offset that grid away from every model's own sampling, returning an empty frame."""
        start = sample_times["past_start"] + offset

        data = read_solar_wind_from_multiple_models(
            start_time=start,
            end_time=sample_times["past_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWENLIL_CME()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert data["model"].notna().any(), "no model contributed any data"
        assert {t.second for t in data.index} == {0}
        assert {t.microsecond for t in data.index} == {0}
        assert data.index.min() == sample_times["past_start"]

    def test_invalid_time_range(self, sample_times):
        with pytest.raises(ValueError):
            read_solar_wind_from_multiple_models(
                start_time=sample_times["future_end"],
                end_time=sample_times["past_start"],
                model_order=[SWOMNI()],
            )

    def test_data_consistency(self, sample_times):
        params = {
            "start_time": sample_times["past_start"],
            "end_time": sample_times["future_start"],
            "model_order": [SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble(), SWENLIL_CME()],
            "historical_data_cutoff_time": sample_times["test_time_now"],
        }

        data1 = read_solar_wind_from_multiple_models(**params)
        data2 = read_solar_wind_from_multiple_models(**params)

        for d1, d2 in zip(data1, data2):
            pd.testing.assert_frame_equal(d1, d2)

    def test_dscovr_value_error_falls_back_to_ace(self, monkeypatch):
        start_time = datetime(2026, 6, 29, 23, 58, tzinfo=timezone.utc)
        end_time = datetime(2026, 6, 30, 0, 1, tzinfo=timezone.utc)
        index = pd.date_range(start_time, end_time, freq="1min", tz="UTC")
        ace_data = pd.DataFrame(
            {
                "speed": [400.0] * len(index),
                "proton_density": [5.0] * len(index),
                "bavg": [7.0] * len(index),
                "temperature": [100000.0] * len(index),
                "bx_gsm": [1.0] * len(index),
                "by_gsm": [2.0] * len(index),
                "bz_gsm": [-1.0] * len(index),
                "file_name": ["ace_file"] * len(index),
            },
            index=index,
        )
        ace_calls = []

        def raise_dscovr_value_error(self, *args, **kwargs):
            raise ValueError("DSCOVR data is only available until 2026-06-29 23:59:59 UTC.")

        def read_ace(self, read_start, read_end, *, download=False, propagation=False):
            ace_calls.append(
                {
                    "model": self,
                    "start_time": read_start,
                    "end_time": read_end,
                    "download": download,
                    "propagation": propagation,
                }
            )
            return ace_data

        monkeypatch.setattr(DSCOVR, "read", raise_dscovr_value_error)
        monkeypatch.setattr(SWACE, "read", read_ace)

        ace_model = SWACE()
        data = read_solar_wind_from_multiple_models(
            start_time=start_time,
            end_time=end_time,
            model_order=[DSCOVR(), ace_model],
            historical_data_cutoff_time=end_time,
            download=True,
        )

        assert len(ace_calls) == 1
        assert ace_calls[0] == {
            "model": ace_model,
            "start_time": start_time,
            "end_time": end_time,
            "download": True,
            "propagation": True,
        }
        assert isinstance(data, pd.DataFrame)
        assert (data["model"] == "ace").all()
        assert (data["file_name"] == "ace_file").all()
        assert data.loc[start_time, "speed"] == 400.0

    def test_omni_empty_dataframe_falls_back_to_ace(self, monkeypatch):
        """Regression test: SWOMNI returning a genuinely empty (0-row) DataFrame
        for the requested range — e.g. no cached/downloadable data available —
        must not be mistaken for "no gaps remain". The reader must still fall
        through to the next model in model_order (SWACE)."""
        start_time = datetime(2026, 7, 24, 10, tzinfo=timezone.utc)
        end_time = datetime(2026, 7, 24, 10, 5, tzinfo=timezone.utc)
        index = pd.date_range(start_time, end_time, freq="1min", tz="UTC")
        ace_data = pd.DataFrame(
            {
                "speed": [400.0] * len(index),
                "proton_density": [5.0] * len(index),
                "bavg": [7.0] * len(index),
                "temperature": [100000.0] * len(index),
                "bx_gsm": [1.0] * len(index),
                "by_gsm": [2.0] * len(index),
                "bz_gsm": [-1.0] * len(index),
                "file_name": ["ace_file"] * len(index),
            },
            index=index,
        )

        def read_empty_omni(self, read_start, read_end, *, download=False):
            return pd.DataFrame(
                columns=["bavg", "bx_gsm", "by_gsm", "bz_gsm", "speed", "proton_density", "temperature", "file_name"]
            )

        def read_ace(self, read_start, read_end, *, download=False, propagation=False):
            return ace_data

        monkeypatch.setattr(SWOMNI, "read", read_empty_omni)
        monkeypatch.setattr(SWACE, "read", read_ace)

        data = read_solar_wind_from_multiple_models(
            start_time=start_time,
            end_time=end_time,
            model_order=[SWOMNI(), SWACE()],
            historical_data_cutoff_time=end_time,
            download=True,
        )

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert (data["model"] == "ace").all()
        assert data.loc[start_time, "speed"] == 400.0

    def test_midl_is_read_as_a_historical_model(self, tmp_path, expected_columns):
        """SWMIDL passed in the model order is accepted and read like the other historical
        models: propagated from L1 and labelled `midl` in the `model` column."""
        start_time = datetime(2024, 11, 22, tzinfo=timezone.utc)
        end_time = datetime(2024, 11, 23, tzinfo=timezone.utc)

        midl_model = SWMIDL(tmp_path / "midl")

        # One MIDL day file per day, plus the preceding day the propagation shift reaches back to.
        # The speed ramps slightly across the window: a strictly constant speed would shift every
        # sample by the same amount and `sw_mag_propagation` would round half the minutes onto
        # each other, leaving an alternating grid that says nothing about the model wiring.
        for day in pd.date_range(start_time - timedelta(days=1), end_time, freq="D"):
            index = pd.date_range(day, day + timedelta(hours=24) - timedelta(minutes=1), freq="1min", tz="UTC")
            index.name = "t"
            speed = np.linspace(400.0, 401.0, len(index))
            df = pd.DataFrame(
                {
                    "bavg": 13.0,
                    "bx_gsm": 3.0,
                    "by_gsm": -4.0,
                    "bz_gsm": 12.0,
                    "proton_density": 5.0,
                    "speed": speed,
                    "temperature": 120000.0,
                    "pdyn": 2e-6 * 5.0 * speed**2,
                },
                index=index,
            )
            path = midl_model.data_dir / day.strftime("%Y/%m") / midl_model._file_name(day, "L1")
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)

        data = read_solar_wind_from_multiple_models(
            start_time=start_time,
            end_time=end_time,
            model_order=[midl_model],
            historical_data_cutoff_time=end_time,
        )

        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in expected_columns)
        assert (data["model"].dropna() == "midl").all()
        assert data["speed"].notna().mean() > 0.99
        assert data["bavg"].dropna().eq(13.0).all()
        assert data["speed"].dropna().between(400.0, 401.0).all()

    def test_imap_is_read_as_a_historical_model(self, tmp_path, expected_columns):
        """SWIMAP passed in the model order is accepted and read like the other historical
        models: propagated from L1 and labelled `imap` in the `model` column."""
        start_time = datetime(2024, 11, 22, tzinfo=timezone.utc)
        end_time = datetime(2024, 11, 23, tzinfo=timezone.utc)

        imap_model = SWIMAP(tmp_path / "imap")

        for day in pd.date_range(start_time - timedelta(days=1), end_time, freq="D"):
            index = pd.date_range(day, day + timedelta(hours=24) - timedelta(minutes=1), freq="1min", tz="UTC")
            index.name = "t"
            speed = np.linspace(400.0, 401.0, len(index))
            df = pd.DataFrame(
                {
                    "bx_gsm": 3.0,
                    "by_gsm": -4.0,
                    "bz_gsm": 12.0,
                    "bavg": 13.0,
                    "speed": speed,
                    "proton_density": 5.0,
                    "temperature": 120000.0,
                    "pdyn": 2e-6 * 5.0 * speed**2,
                },
                index=index,
            )
            path = imap_model.data_dir / day.strftime("%Y/%m") / f"IMAP_SW_NOWCAST_{day.strftime('%Y%m%d')}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)

        data = read_solar_wind_from_multiple_models(
            start_time=start_time,
            end_time=end_time,
            model_order=[imap_model],
            historical_data_cutoff_time=end_time,
        )

        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in expected_columns)
        assert (data["model"].dropna() == "imap").all()
        assert data["speed"].notna().mean() > 0.99
        assert data["bavg"].dropna().eq(13.0).all()
        assert data["speed"].dropna().between(400.0, 401.0).all()

    def test_model_check_with_wrong_class(self, sample_times):
        class FakeModel:
            pass

        fake = FakeModel()
        with pytest.raises(ModelError, match="Unknown or incompatible model"):
            read_solar_wind_from_multiple_models(
                start_time=sample_times["past_start"],
                end_time=sample_times["future_end"],
                model_order=[fake],
            )

    @pytest.mark.parametrize(("fill_average"), [False, True])
    def test_fill_modes_do_not_truncate_final_dataframe(self, monkeypatch, fill_average):
        start_time = datetime(2024, 11, 25, 0, 0, tzinfo=timezone.utc)
        historical_data_cutoff_time = start_time + timedelta(minutes=5)
        end_time = start_time + timedelta(minutes=10)
        index = pd.date_range(start_time, end_time, freq="1min", tz="UTC")
        data = pd.DataFrame(
            {
                "speed": [400.0] * 6 + [np.nan] * 5,
                "model": ["omni"] * 6 + [None] * 5,
                "file_name": ["test_file.txt"] * 6 + [None] * 5,
            },
            index=index,
        )

        monkeypatch.setattr(READ_SW_MODULE, "_read_from_model", lambda *args, **kwargs: data)

        result = read_solar_wind_from_multiple_models(
            start_time=start_time,
            end_time=end_time,
            model_order=[SWOMNI(), SWSWIFTEnsemble(), SWENLIL_CME()],
            historical_data_cutoff_time=historical_data_cutoff_time,
            fill_average=fill_average,
        )

        assert result.index.max() == end_time

    def test_average_fill_uses_expected_values(self, monkeypatch):
        start_time = datetime(2024, 11, 25, 0, 0, tzinfo=timezone.utc)
        historical_data_cutoff_time = start_time + timedelta(minutes=5)
        end_time = start_time + timedelta(minutes=10)
        index = pd.date_range(start_time, end_time, freq="1min", tz="UTC")
        average_values = AVERAGE_VALUES_TO_FILL
        data = pd.DataFrame(
            {
                **{col: [1.0] * 6 + [np.nan] * 5 for col in average_values},
                "model": ["omni"] * 6 + [None] * 5,
                "file_name": ["test_file.txt"] * 6 + [None] * 5,
            },
            index=index,
        )

        monkeypatch.setattr(READ_SW_MODULE, "_read_from_model", lambda *args, **kwargs: data)

        result = read_solar_wind_from_multiple_models(
            start_time=start_time,
            end_time=end_time,
            model_order=[SWOMNI()],
            historical_data_cutoff_time=historical_data_cutoff_time,
            fill_average=True,
        )

        future_mask = result.index > historical_data_cutoff_time
        assert result.index.max() == end_time
        for col, avg_value in average_values.items():
            assert result.loc[historical_data_cutoff_time, col] == 1.0
            np.testing.assert_allclose(result.loc[future_mask, col].to_numpy(), avg_value)
        assert (result.loc[future_mask, "model"] == "10_year_average_filled").all()
        assert (result.loc[future_mask, "file_name"] == "10_year_average_filled").all()

    def test_3_hour_interpolation(self, sample_times, expected_columns):
        # Use a longer time range to increase chances of gaps that need interpolation
        extended_start = sample_times["past_start"] - timedelta(days=2)
        extended_end = sample_times["past_end"] + timedelta(days=1)

        data_no_rec = read_solar_wind_from_multiple_models(
            start_time=extended_start,
            end_time=extended_end,
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWENLIL_CME()],
            historical_data_cutoff_time=sample_times["test_time_now"],
            download=False,
            do_interpolation=True,
        )

        data_with_rec = read_solar_wind_from_multiple_models(
            start_time=extended_start,
            end_time=extended_end,
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWENLIL_CME()],
            historical_data_cutoff_time=sample_times["test_time_now"],
            download=True,
            do_interpolation=True,
        )

        nan_count_no_rec = data_no_rec.isna().sum()
        nan_count_with_rec = data_with_rec.isna().sum()

        for i in expected_columns:
            assert nan_count_with_rec[i] <= nan_count_no_rec[i]

    def test_ensemble_reduction_methods(self, sample_times, expected_columns):
        reduction_methods = [None, "mean", "median"]

        for method in reduction_methods:
            data = read_solar_wind_from_multiple_models(
                start_time=sample_times["future_start"],
                end_time=sample_times["future_end"],
                model_order=[SWSWIFTEnsemble()],
                reduce_ensemble=method,
                historical_data_cutoff_time=sample_times["test_time_now"],
            )

            if method is None:
                assert isinstance(data, list)
                assert len(data) > 1
                for d in data:
                    assert all(col in d.columns for col in expected_columns)
            else:
                assert isinstance(data, pd.DataFrame)
                assert all(col in data.columns for col in expected_columns)

    def test_cme_runs_become_ensembles_without_swift(self, enlil_instance, sample_times):
        now = sample_times["test_time_now"]
        for run_time in ["0300", "0900", "1800"]:
            _write_enlil_run(enlil_instance, "cme", now, run_time, 400.0)
        _write_enlil_run(enlil_instance, "bkg", now, "0000", 300.0)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(days=2),
            model_order=[SWOMNI(), enlil_instance],
            historical_data_cutoff_time=now,
        )

        assert len(data) == 3
        for d in data:
            assert (d.loc[d["model"] == "enlil_cme", "speed"] == 400.0).all()
        assert len({d.loc[d["model"] == "enlil_cme", "file_name"].iloc[0] for d in data}) == 3

    @pytest.mark.parametrize(("mode", "expected_label"), [("cme", "enlil_cme"), ("bkg", "enlil_bkg")])
    def test_model_label_names_the_run_mode(
        self, enlil_instance, enlil_bkg_instance, sample_times, mode, expected_label
    ):
        now = sample_times["test_time_now"]
        model = enlil_instance if mode == "cme" else enlil_bkg_instance
        _write_enlil_run(model, mode, now, "1800" if mode == "cme" else "0000", 400.0)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(days=2),
            model_order=[SWOMNI(), model],
            historical_data_cutoff_time=now,
        )

        labelled = data[data["model"] == expected_label]
        assert not labelled.empty
        # every labelled row names the file it came from, and that file is of the labelled mode
        assert labelled["file_name"].notna().all()
        assert labelled["file_name"].str.contains(f"_{mode}_").all()
        assert "enlil" not in data["model"].dropna().unique()

    def test_model_and_file_name_agree_everywhere(self, enlil_instance, sample_times):
        """A row carries a model label exactly when it carries the file that produced it."""
        now = sample_times["test_time_now"]
        _write_enlil_run(enlil_instance, "cme", now, "1800", 403.0, days=4)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=3),
            end_time=now + timedelta(days=6),
            model_order=[SWOMNI(), enlil_instance, SWSWIFTEnsemble()],
            historical_data_cutoff_time=now,
        )

        for d in data:
            pd.testing.assert_series_equal(
                d["model"].notna(), d["file_name"].notna(), check_names=False, check_dtype=False
            )
            for label, group in d.dropna(subset=["model"]).groupby("model"):
                marker = {"omni": "OMNI", "swift": "gsm_", "enlil_cme": "_cme_"}[str(label)]
                assert group["file_name"].astype(str).str.contains(marker).all()

    def test_falls_back_to_bkg_when_no_cme_runs(self, enlil_instance, enlil_bkg_instance, sample_times):
        now = sample_times["test_time_now"]
        _write_enlil_run(enlil_bkg_instance, "bkg", now, "0000", 300.0)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(days=2),
            model_order=[SWOMNI(), enlil_instance, enlil_bkg_instance],
            historical_data_cutoff_time=now,
        )

        assert isinstance(data, pd.DataFrame)
        assert (data.loc[data["model"] == "enlil_bkg", "speed"] == 300.0).all()
        assert data.loc[data["model"] == "enlil_bkg", "file_name"].str.contains("bkg").all()

    def test_bkg_class_in_model_order_reads_only_background_runs(self, tmp_path, sample_times):
        """Either run-mode class can go in the model order, and SWENLIL_BKG keeps to the
        background run even on a date that also has CME runs."""
        now = sample_times["test_time_now"]
        bkg_instance = SWENLIL_BKG(tmp_path / "enlil")
        _write_enlil_run(bkg_instance, "bkg", now, "0000", 300.0)
        _write_enlil_run(bkg_instance, "cme", now, "1800", 403.0)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(days=2),
            model_order=[SWOMNI(), bkg_instance],
            historical_data_cutoff_time=now,
        )

        assert isinstance(data, pd.DataFrame)
        assert "enlil_cme" not in data["model"].dropna().unique()
        assert (data.loc[data["model"] == "enlil_bkg", "speed"] == 300.0).all()

    def test_swift_hands_over_to_cme_then_to_bkg(self, enlil_instance, enlil_bkg_instance, sample_times):
        """The forecast window is covered in model order: SWIFT for as far as it reaches, then the
        latest CME run, then the background run for whatever the CME run does not cover."""
        now = sample_times["test_time_now"]
        # The SWIFT ensemble reaches 2024-12-02 (now + 7 days), so these horizons make each model
        # run out in turn before end_time.
        _write_enlil_run(enlil_instance, "cme", now, "1800", 403.0, days=8)
        _write_enlil_run(enlil_bkg_instance, "bkg", now, "0000", 300.0, days=12)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=1),
            end_time=sample_times["far_future_end"],
            model_order=[SWOMNI(), SWSWIFTEnsemble(), enlil_instance, enlil_bkg_instance],
            historical_data_cutoff_time=now,
        )

        assert isinstance(data, list)
        for d in data:
            swift = d.index[d["model"] == "swift"]
            cme = d.index[d["model"] == "enlil_cme"]
            bkg = d.index[d["model"] == "enlil_bkg"]

            assert len(swift) > 0 and len(cme) > 0 and len(bkg) > 0
            assert cme.min() > swift.max()
            assert bkg.min() > cme.max()
            assert (d.loc[d["model"] == "enlil_cme", "speed"] == 403.0).all()
            assert (d.loc[d["model"] == "enlil_bkg", "speed"] == 300.0).all()

    def test_average_fill_closes_the_tail_neither_enlil_reaches(self, enlil_instance, enlil_bkg_instance, sample_times):
        now = sample_times["test_time_now"]
        end_time = now + timedelta(days=6)
        _write_enlil_run(enlil_instance, "cme", now, "1800", 403.0, days=2)
        _write_enlil_run(enlil_bkg_instance, "bkg", now, "0000", 300.0, days=4)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=1),
            end_time=end_time,
            model_order=[SWOMNI(), enlil_instance, enlil_bkg_instance],
            historical_data_cutoff_time=now,
            fill_average=True,
        )

        assert isinstance(data, pd.DataFrame)
        assert data.index.max() == end_time
        # the forecast window is covered end to end; the historical part keeps OMNI's own gaps
        assert not data.loc[data.index > now, "speed"].isna().any()

        filled = data.index[data["model"] == "10_year_average_filled"]
        assert len(filled) > 0
        # the averages only start once the background run, itself after the CME run, runs out
        assert filled.min() > data.index[data["model"] == "enlil_bkg"].max()
        assert data.index[data["model"] == "enlil_bkg"].min() > data.index[data["model"] == "enlil_cme"].max()

    def test_walks_back_to_an_earlier_run_date(self, enlil_instance, sample_times):
        now = sample_times["test_time_now"]
        _write_enlil_run(enlil_instance, "cme", now - timedelta(days=2), "1200", 500.0)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(days=2),
            model_order=[SWOMNI(), enlil_instance],
            historical_data_cutoff_time=now,
        )

        assert (data.loc[data["model"] == "enlil_cme", "speed"] == 500.0).all()
        assert data.loc[data["model"] == "enlil_cme", "file_name"].str.contains("20241123").all()

    def test_latest_cme_run_is_copied_across_swift_members(self, enlil_instance, sample_times):
        now = sample_times["test_time_now"]
        for run_time, value in [("0300", 401.0), ("0900", 402.0), ("1800", 403.0)]:
            _write_enlil_run(enlil_instance, "cme", now, run_time, value)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=3),
            end_time=now + timedelta(days=9),
            model_order=[SWOMNI(), SWSWIFTEnsemble(), enlil_instance],
            historical_data_cutoff_time=now,
        )

        assert len(data) > 1
        for d in data:
            enlil_rows = d[d["model"] == "enlil_cme"]
            swift_rows = d[d["model"] == "swift"]
            assert not enlil_rows.empty
            # only the latest run of the day is used once SWIFT has fixed the ensemble size
            assert (enlil_rows["speed"] == 403.0).all()
            # ENLIL picks up exactly where SWIFT stops
            assert enlil_rows.index.min() > swift_rows.index.max()

    def test_enlil_never_fills_before_the_historical_cutoff(self, enlil_bkg_instance, sample_times):
        now = sample_times["test_time_now"]
        _write_enlil_run(enlil_bkg_instance, "bkg", now - timedelta(days=3), "0000", 300.0)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=3),
            end_time=now + timedelta(days=2),
            model_order=[SWOMNI(), enlil_bkg_instance],
            historical_data_cutoff_time=now,
        )

        assert (data[data["model"] == "enlil_bkg"].index > now).all()

    def test_no_enlil_data_leaves_forecast_window_empty(self, enlil_instance, sample_times):
        now = sample_times["test_time_now"]
        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(days=2),
            model_order=[SWOMNI(), enlil_instance],
            historical_data_cutoff_time=now,
        )

        assert isinstance(data, pd.DataFrame)
        assert not data["model"].dropna().str.startswith("enlil").any()
        assert data.index.max() <= now

    def test_cme_runs_become_ensembles_when_swift_is_present_but_has_no_data(self, enlil_instance):
        """SWIFT owns the ensemble only when it actually supplies one. On a day the SWIFT run did
        not happen, ENLIL's CME runs become the ensemble rather than collapsing to a single run."""
        # 2024-11-20 has no SWIFT ensemble folder in the test data, unlike 2024-11-25.
        quiet_day = datetime(2024, 11, 20, tzinfo=timezone.utc)
        for run_time, value in [("0300", 401.0), ("0900", 402.0), ("1800", 403.0)]:
            _write_enlil_run(enlil_instance, "cme", quiet_day, run_time, value, days=4)

        data = read_solar_wind_from_multiple_models(
            start_time=quiet_day - timedelta(days=1),
            end_time=quiet_day + timedelta(days=3),
            model_order=[SWOMNI(), SWSWIFTEnsemble(), enlil_instance],
            historical_data_cutoff_time=quiet_day,
        )

        assert isinstance(data, list)
        assert len(data) == 3
        # one member per CME run, each keeping its own run's values and file
        speeds = [d.loc[d["model"] == "enlil_cme", "speed"].unique().tolist() for d in data]
        assert speeds == [[401.0], [402.0], [403.0]]
        assert "swift" not in data[0]["model"].dropna().unique()

    def test_enlil_before_swift_uses_latest_run_and_swift_fills_the_tail(self, enlil_instance, sample_times):
        now = sample_times["test_time_now"]
        # ENLIL's horizon is deliberately shorter than SWIFT's so SWIFT has a tail to fill.
        for run_time, value in [("0300", 401.0), ("1800", 403.0)]:
            _write_enlil_run(enlil_instance, "cme", now, run_time, value, days=4)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=3),
            end_time=now + timedelta(days=6),
            model_order=[SWOMNI(), enlil_instance, SWSWIFTEnsemble()],
            historical_data_cutoff_time=now,
        )

        assert len(data) > 1
        for d in data:
            enlil_rows = d[d["model"] == "enlil_cme"]
            swift_rows = d[d["model"] == "swift"]
            # SWIFT owns the ensemble, so ENLIL contributes its latest run only
            assert (enlil_rows["speed"] == 403.0).all()
            # ENLIL is read first, so it leads and SWIFT picks up the remainder
            assert enlil_rows.index.min() > now
            assert not swift_rows.empty
            assert swift_rows.index.min() > enlil_rows.index.max()

    def test_enlil_first_with_no_historical_model(self, enlil_instance, sample_times):
        """ENLIL populating the frame first must not lock the file_name column to a string dtype:
        SWIFT carries its file names as Path and has to be able to write into the same column."""
        now = sample_times["test_time_now"]
        _write_enlil_run(enlil_instance, "cme", now, "1800", 403.0, days=4)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=3),
            end_time=now + timedelta(days=6),
            model_order=[enlil_instance, SWSWIFTEnsemble()],
            historical_data_cutoff_time=now,
        )

        for d in data:
            assert (d.loc[d["model"] == "enlil_cme", "speed"] == 403.0).all()
            assert not d[d["model"] == "swift"].empty

    def test_swift_members_stay_distinct_when_enlil_leads(self, enlil_instance, sample_times):
        now = sample_times["test_time_now"]
        _write_enlil_run(enlil_instance, "cme", now, "1800", 403.0, days=4)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=3),
            end_time=now + timedelta(days=6),
            model_order=[SWOMNI(), enlil_instance, SWSWIFTEnsemble()],
            historical_data_cutoff_time=now,
        )

        swift_speeds = [d.loc[d["model"] == "swift", "speed"].to_numpy() for d in data]
        assert len({s.tobytes() for s in swift_speeds}) > 1

    def test_value_error_from_enlil_is_not_swallowed_as_a_dscovr_fallback(
        self, enlil_instance, sample_times, monkeypatch
    ):
        """Only DSCOVR has an ACE fallback. A ValueError from another model must surface rather
        than be reported as a DSCOVR failure and answered with ACE data from some other path."""
        now = sample_times["test_time_now"]

        def raise_value_error(self, *args, **kwargs):
            raise ValueError("enlil_instance is unhappy")

        monkeypatch.setattr(SWENLIL_CME, "read", raise_value_error)

        with pytest.raises(ValueError, match="enlil_instance is unhappy"):
            read_solar_wind_from_multiple_models(
                start_time=now - timedelta(days=1),
                end_time=now + timedelta(days=2),
                model_order=[SWOMNI(), enlil_instance],
                historical_data_cutoff_time=now,
            )

    def test_unsorted_run_file_is_read_in_time_order(self, enlil_instance, sample_times):
        """A run file whose rows are out of time order must not reach np.interp unsorted, which
        would silently return interpolated nonsense rather than fail."""
        now = sample_times["test_time_now"]
        path = enlil_instance.data_dir / now.strftime("%Y/%m") / f"ENLIL_FORECAST_cme_{now:%Y%m%d}_1800.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        index = pd.DatetimeIndex([now + timedelta(hours=i) for i in range(12)], name="t")
        df = pd.DataFrame({col: np.arange(12, dtype=float) * 100 for col in ENLIL_COLUMNS}, index=index)
        df["file_name"] = str(path)
        shuffled = list(range(12))
        shuffled[3], shuffled[8] = shuffled[8], shuffled[3]
        df.iloc[shuffled].to_csv(path)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(hours=11),
            model_order=[SWOMNI(), enlil_instance],
            historical_data_cutoff_time=now,
        )

        speed = data.loc[data["model"] == "enlil_cme", "speed"]
        for hour in range(1, 12):
            assert speed.loc[now + timedelta(hours=hour)] == pytest.approx(hour * 100.0)

    def test_reduce_ensemble_collapses_cme_runs(self, enlil_instance, sample_times):
        now = sample_times["test_time_now"]
        for run_time, value in [("0300", 400.0), ("0900", 500.0), ("1800", 600.0)]:
            _write_enlil_run(enlil_instance, "cme", now, run_time, value)

        data = read_solar_wind_from_multiple_models(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(days=2),
            model_order=[SWOMNI(), enlil_instance],
            reduce_ensemble="mean",
            historical_data_cutoff_time=now,
        )

        assert isinstance(data, pd.DataFrame)
        assert (data.loc[data["model"] == "enlil_cme", "speed"] == 500.0).all()


class TestInterpolateShortGaps:
    @pytest.fixture
    def sample_dataframe_with_short_gap(self):
        np.random.seed(42)
        times = pd.date_range("2024-01-01 00:00:00", periods=480, freq="1min", tz=timezone.utc)
        n_points = len(times)
        data = {
            "proton_density": np.linspace(5.0, 15.0, n_points) + np.random.normal(0, 0.5, n_points),
            "speed": np.linspace(400.0, 600.0, n_points) + np.random.normal(0, 20, n_points),
            "bavg": np.linspace(8.0, 18.0, n_points) + np.random.normal(0, 1, n_points),
            "temperature": np.linspace(80000, 120000, n_points) + np.random.normal(0, 5000, n_points),
            "bx_gsm": np.sin(np.linspace(0, 4 * np.pi, n_points)) * 5 + np.random.normal(0, 0.5, n_points),
            "by_gsm": np.cos(np.linspace(0, 3 * np.pi, n_points)) * 3 + np.random.normal(0, 0.3, n_points),
            "bz_gsm": np.linspace(-5, 5, n_points) + np.random.normal(0, 1, n_points),
            "model": ["omni"] * n_points,
            "file_name": ["test_file.txt"] * n_points,
        }
        df = pd.DataFrame(data, index=times)

        gap_start = 150
        gap_end = 270
        numeric_cols = ["proton_density", "speed", "bavg", "temperature", "bx_gsm", "by_gsm", "bz_gsm"]
        df.loc[df.index[gap_start:gap_end], numeric_cols] = np.nan

        return df

    @pytest.fixture
    def sample_dataframe_with_long_gap(self):
        np.random.seed(123)
        times = pd.date_range("2024-01-01 00:00:00", periods=600, freq="1min", tz=timezone.utc)

        n_points = len(times)
        data = {
            "proton_density": np.linspace(3.0, 12.0, n_points) + np.random.normal(0, 0.8, n_points),
            "speed": np.linspace(350.0, 550.0, n_points) + np.random.normal(0, 25, n_points),
            "bavg": np.linspace(6.0, 16.0, n_points) + np.random.normal(0, 1.2, n_points),
            "temperature": np.linspace(70000, 130000, n_points) + np.random.normal(0, 8000, n_points),
            "bx_gsm": np.sin(np.linspace(0, 6 * np.pi, n_points)) * 4 + np.random.normal(0, 0.7, n_points),
            "by_gsm": np.cos(np.linspace(0, 4 * np.pi, n_points)) * 2.5 + np.random.normal(0, 0.4, n_points),
            "bz_gsm": np.linspace(-8, 8, n_points) + np.random.normal(0, 1.5, n_points),
            "model": ["omni"] * n_points,
            "file_name": ["test_file.txt"] * n_points,
        }
        df = pd.DataFrame(data, index=times)

        gap_start = 180
        gap_end = 420
        numeric_cols = ["proton_density", "speed", "bavg", "temperature", "bx_gsm", "by_gsm", "bz_gsm"]
        df.loc[df.index[gap_start:gap_end], numeric_cols] = np.nan

        return df

    @pytest.fixture
    def sample_dataframe_no_gaps(self):
        np.random.seed(456)

        times = pd.date_range("2024-01-01 00:00:00", periods=120, freq="1min", tz=timezone.utc)

        n_points = len(times)
        data = {
            "proton_density": np.linspace(4.0, 8.0, n_points) + np.random.normal(0, 0.3, n_points),
            "speed": np.linspace(380.0, 480.0, n_points) + np.random.normal(0, 15, n_points),
            "bavg": np.linspace(9.0, 13.0, n_points) + np.random.normal(0, 0.8, n_points),
            "temperature": np.linspace(85000, 105000, n_points) + np.random.normal(0, 3000, n_points),
            "bx_gsm": np.sin(np.linspace(0, 2 * np.pi, n_points)) * 3 + np.random.normal(0, 0.4, n_points),
            "by_gsm": np.cos(np.linspace(0, np.pi, n_points)) * 2 + np.random.normal(0, 0.2, n_points),
            "bz_gsm": np.linspace(-3, 3, n_points) + np.random.normal(0, 0.8, n_points),
            "model": ["omni"] * n_points,
            "file_name": ["test_file.txt"] * n_points,
        }
        df = pd.DataFrame(data, index=times)
        return df

    def test_interpolate_short_gap_success(self, sample_dataframe_with_short_gap):
        gap_start, gap_end = 150, 270
        original_before = sample_dataframe_with_short_gap["proton_density"].iloc[gap_start - 1]
        original_after = sample_dataframe_with_short_gap["proton_density"].iloc[gap_end]

        result = _interpolate_short_gaps(sample_dataframe_with_short_gap, max_gap_minutes=180)

        gap_slice = slice(gap_start, gap_end)
        assert not result["proton_density"].iloc[gap_slice].isna().any()
        assert not result["speed"].iloc[gap_slice].isna().any()
        assert not result["bavg"].iloc[gap_slice].isna().any()

        assert all(result["file_name"].iloc[gap_slice] == "interpolated")
        assert all(result["model"].iloc[gap_slice] == "interpolated")

        assert result["proton_density"].iloc[0] == sample_dataframe_with_short_gap["proton_density"].iloc[0]
        assert result["proton_density"].iloc[gap_start - 1] == original_before
        assert result["proton_density"].iloc[gap_end] == original_after

    def test_long_gap_not_interpolated(self, sample_dataframe_with_long_gap):
        result = _interpolate_short_gaps(sample_dataframe_with_long_gap, max_gap_minutes=180)

        gap_start, gap_end = 180, 420
        gap_slice = slice(gap_start, gap_end)
        assert result["proton_density"].iloc[gap_slice].isna().all()
        assert result["speed"].iloc[gap_slice].isna().all()
        assert result["bavg"].iloc[gap_slice].isna().all()

        assert result["file_name"].iloc[gap_start] == "test_file.txt"

    def test_no_gaps_unchanged(self, sample_dataframe_no_gaps):
        result = _interpolate_short_gaps(sample_dataframe_no_gaps)

        pd.testing.assert_frame_equal(result, sample_dataframe_no_gaps)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        result = _interpolate_short_gaps(empty_df)

        assert result.empty
        pd.testing.assert_frame_equal(result, empty_df)


class TestRecursiveFill27dHistorical:
    @pytest.fixture(scope="session", autouse=True)
    def set_env_var(self):
        ENV_VAR_NAMES = {
            "OMNI_HIGH_RES_STREAM_DIR": f"{str(DATA_DIR)}",
            "RT_SW_ACE_STREAM_DIR": f"{str(DATA_DIR)}/ACE_RT",
            "SW_DSCOVR_STREAM_DIR": f"{str(DATA_DIR)}/DSCOVR",
        }

        for key, var in ENV_VAR_NAMES.items():
            os.environ[key] = ENV_VAR_NAMES[key]

    @pytest.fixture
    def sample_dataframe_with_gaps(self):
        current_time = datetime(2024, 11, 25, tzinfo=timezone.utc)
        times = pd.date_range(current_time, periods=1440, freq="1min", tz=timezone.utc)

        data = {
            "proton_density": [np.nan] * 1440,
            "speed": [np.nan] * 1440,
            "bavg": [np.nan] * 1440,
            "temperature": [np.nan] * 1440,
            "bx_gsm": [np.nan] * 1440,
            "by_gsm": [np.nan] * 1440,
            "bz_gsm": [np.nan] * 1440,
            "model": [None] * 1440,
            "file_name": [None] * 1440,
        }
        df = pd.DataFrame(data, index=times)
        return df

    @pytest.fixture
    def sample_dataframe_no_gaps(self):
        current_time = datetime(2024, 11, 25, tzinfo=timezone.utc)
        times = pd.date_range(current_time, periods=5, freq="1min", tz=timezone.utc)

        data = {
            "proton_density": [5.0, 6.0, 7.0, 8.0, 9.0],
            "speed": [400.0, 410.0, 420.0, 430.0, 440.0],
            "bavg": [10.0, 11.0, 12.0, 13.0, 14.0],
            "model": ["omni"] * 5,
            "file_name": ["test_file.txt"] * 5,
        }
        df = pd.DataFrame(data, index=times)
        return df
