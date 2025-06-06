import datetime as dt
from pathlib import Path
import pytest

from data_management.io.RBMDataSet import RBMDataSetManager
from data_management.io.RBMDataSet import SatelliteEnum, InstrumentEnum, MfmEnum


@pytest.fixture
def manager_args():
    return {
        "start_time": dt.datetime(2025, 4, 1, tzinfo=dt.timezone.utc),
        "end_time": dt.datetime(2025, 4, 30, tzinfo=dt.timezone.utc),
        "folder_path": Path("/mock/path"),
        "instrument": InstrumentEnum.MAGED,
        "mfm": MfmEnum.T89,
    }


def test_singleton_prevents_direct_init():
    with pytest.raises(RuntimeError):
        _ = RBMDataSetManager()


def test_single_satellite_returns_dataset(manager_args):
    dataset = RBMDataSetManager.load(
        satellite=SatelliteEnum.GOESSecondary,
        **manager_args,
    )
    assert dataset.get_satellite_name() == "secondary"


def test_same_parameters_return_same_instance(manager_args):
    ds1 = RBMDataSetManager.load(satellite=SatelliteEnum.GOESSecondary, **manager_args)
    ds2 = RBMDataSetManager.load(satellite=SatelliteEnum.GOESSecondary, **manager_args)
    assert ds1 is ds2


def test_different_satellite_returns_different_instance(manager_args):
    ds1 = RBMDataSetManager.load(satellite=SatelliteEnum.GOESPrimary, **manager_args)
    ds2 = RBMDataSetManager.load(satellite=SatelliteEnum.GOESSecondary, **manager_args)
    assert ds1 is not ds2


def test_string_input_for_satellite(manager_args):
    dataset = RBMDataSetManager.load(satellite="GOESSecondary", **manager_args)
    assert dataset.get_satellite_name() == "secondary"


def test_multiple_satellites_returns_list(manager_args):
    datasets = RBMDataSetManager.load(
        satellite=[SatelliteEnum.GOESPrimary, SatelliteEnum.GOESSecondary],
        **manager_args,
    )
    assert isinstance(datasets, list)
    assert len(datasets) == 2
    assert all(isinstance(ds, type(datasets[0])) for ds in datasets)
