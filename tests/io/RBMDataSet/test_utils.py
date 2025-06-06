import pickle
import numpy as np
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from data_management.io.RBMDataSet import utils
import scipy


def test_join_var():
    a = np.array([1, 2])
    b = np.array([3, 4])
    result = utils.join_var(a, b)
    np.testing.assert_array_equal(result, [1, 2, 3, 4])


def test_get_file_path_any_format_single_match(tmp_path: Path):
    file = tmp_path / "file.pickle"
    file.touch()
    result = utils.get_file_path_any_format(tmp_path, "file", "pickle")
    assert result == file


def test_get_file_path_any_format_multiple_match(tmp_path: Path):
    (tmp_path / "file.pickle").touch()
    (tmp_path / "file.mat").touch()
    result_pickle = utils.get_file_path_any_format(tmp_path, "file", "pickle")
    result_mat = utils.get_file_path_any_format(tmp_path, "file", "mat")
    assert result_pickle.name == "file.pickle"
    assert result_mat.name == "file.mat"


def test_get_file_path_any_format_no_match(tmp_path: Path):
    (tmp_path / "nonexistent.tmp").touch()
    result = utils.get_file_path_any_format(tmp_path, "nonexistent", "pickle")
    assert result is None

def test_get_file_path_any_format_no_file(tmp_path: Path):
    result = utils.get_file_path_any_format(tmp_path, "nonexistent", "pickle")
    assert result is None


def test_load_file_any_format_pickle(tmp_path: Path):
    file = tmp_path / "data.pickle"
    content = {"a": np.array([1, 2])}
    with file.open("wb") as f:
        pickle.dump(content, f)

    loaded = utils.load_file_any_format(file)
    np.testing.assert_array_equal(loaded["a"], content["a"])


def test_load_file_any_format_mat(tmp_path: Path):
    file = tmp_path / "data.mat"
    content = {"a": np.array([1, 2])}
    with file.open("wb") as f:
        scipy.io.savemat(f.name, content)

    loaded = utils.load_file_any_format(file)
    np.testing.assert_array_equal(loaded["a"], content["a"])

def test_load_file_any_format_invalid_extension(tmp_path: Path):
    file = tmp_path / "file.unknown"
    file.touch()
    with pytest.raises(NotImplementedError):
        utils.load_file_any_format(file)


def test_round_seconds():
    dt1 = datetime(2024, 1, 1, 12, 0, 0, 600_000)
    dt2 = datetime(2024, 1, 1, 12, 0, 0, 300_000)
    assert utils.round_seconds(dt1).second == 1
    assert utils.round_seconds(dt2).second == 0


def test_python2matlab_and_matlab2python_roundtrip():
    dt1 = datetime(2024, 4, 16, 15, 30, 0, tzinfo=timezone.utc)
    matlab_time = utils.python2matlab(dt1)
    dt2 = utils.matlab2python(matlab_time)
    assert dt2.year == dt1.year
    assert dt2.month == dt1.month
    assert dt2.day == dt1.day


def test_matlab2python_iterable():
    dt1 = datetime(2024, 4, 16, 15, 30, 0, tzinfo=timezone.utc)
    matlab_time = utils.python2matlab(dt1)
    dt2 = utils.matlab2python([matlab_time])[0]
    assert isinstance(dt2, datetime)
    assert dt2.year == dt1.year


def test_pol2cart_and_cart2pol():
    theta = np.array([0, np.pi / 2, np.pi])
    radius = np.array([1, 1, 1])
    x, y = utils.pol2cart(theta, radius)
    theta2, r2 = utils.cart2pol(x, y)
    np.testing.assert_allclose(theta % (2 * np.pi), theta2 % (2 * np.pi), atol=1e-5)
    np.testing.assert_allclose(radius, r2, atol=1e-5)
