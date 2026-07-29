# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from swvo.io.base import BaseIO


class TestBaseIO(BaseIO):
    """Implementation of BaseIO for testing."""

    ENV_VAR_NAME = "TEST_DATA_DIR"
    LABEL = "test_label"

    def read(self, *args, **kwargs):
        """Implementation of read method."""
        return pd.DataFrame({"data": [1, 2, 3]})

    def download_and_process(self, *args, **kwargs):
        """Implementation of download_and_process method."""
        pass


class TestBaseIOInitialization:
    """Test BaseIO initialization with various configurations."""

    def test_initialization_with_data_dir(self, tmp_path):
        """Test initialization with explicitly provided data_dir."""
        io = TestBaseIO(data_dir=tmp_path)
        assert io.data_dir == tmp_path
        assert io.data_dir.exists()

    def test_initialization_creates_directory(self, tmp_path):
        """Test that initialization creates data directory if it doesn't exist."""
        data_dir = tmp_path / "new_dir" / "nested"
        assert not data_dir.exists()

        io = TestBaseIO(data_dir=data_dir)

        assert data_dir.exists()
        assert io.data_dir == data_dir

    def test_initialization_with_env_var(self, tmp_path):
        """Test initialization using environment variable."""
        env_dir = tmp_path / "env_data"
        with patch.dict(os.environ, {"TEST_DATA_DIR": str(env_dir)}):
            io = TestBaseIO()
            assert io.data_dir == env_dir
            assert env_dir.exists()

    def test_initialization_prefer_env_var_true(self, tmp_path):
        """Test that prefer_env_var=True prioritizes environment variable."""
        env_dir = tmp_path / "env_data"
        passed_dir = tmp_path / "passed_data"

        with patch.dict(os.environ, {"TEST_DATA_DIR": str(env_dir)}):
            io = TestBaseIO(data_dir=passed_dir, prefer_env_var=True)
            assert io.data_dir == env_dir

    def test_initialization_prefer_env_var_false(self, tmp_path):
        """Test that prefer_env_var=False uses passed data_dir when available."""
        env_dir = tmp_path / "env_data"
        passed_dir = tmp_path / "passed_data"

        with patch.dict(os.environ, {"TEST_DATA_DIR": str(env_dir)}):
            io = TestBaseIO(data_dir=passed_dir, prefer_env_var=False)
            assert io.data_dir == passed_dir

    def test_initialization_no_data_dir_no_env_var(self):
        """Test that ValueError is raised when no data_dir and no env_var."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Necessary environment variable TEST_DATA_DIR not set!"):
                TestBaseIO()

    def test_initialization_no_data_dir_missing_env_var(self):
        """Test ValueError when data_dir is None and ENV_VAR_NAME not in os.environ."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Necessary environment variable TEST_DATA_DIR not set!"):
                TestBaseIO(data_dir=None)

    def test_initialization_prefer_env_var_without_env_var_set(self, tmp_path):
        """Test that passed data_dir is used when prefer_env_var=True but env var not set.

        When prefer_env_var=True and env var is not set, the fallback is to use
        the passed data_dir (doesn't raise error if data_dir is provided).
        """
        passed_dir = tmp_path / "passed_data"

        with patch.dict(os.environ, {}, clear=True):
            io = TestBaseIO(data_dir=passed_dir, prefer_env_var=True)
            assert io.data_dir == passed_dir

    def test_initialization_logs_message(self, tmp_path, caplog):
        """Test that initialization logs the data directory."""
        with caplog.at_level(logging.INFO):
            TestBaseIO(data_dir=tmp_path)

        assert "TestBaseIO data directory:" in caplog.text
        assert str(tmp_path) in caplog.text


class TestBaseIOAbstractMethods:
    """Test abstract method behavior."""

    def test_baseio_cannot_be_instantiated_directly(self):
        """Test that BaseIO cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseIO()

    def test_baseio_implementation_must_implement_read(self):
        """Test that subclass must implement read method."""

        class IncompleteIO(BaseIO):
            ENV_VAR_NAME = "TEST_DATA_DIR"
            LABEL = "test"

            def download_and_process(self, *args, **kwargs):
                pass

        with pytest.raises(TypeError):
            IncompleteIO(data_dir=Path("/tmp"))

    def test_baseio_implementation_must_implement_download_and_process(self):
        """Test that subclass must implement download_and_process method."""

        class IncompleteIO(BaseIO):
            ENV_VAR_NAME = "TEST_DATA_DIR"
            LABEL = "test"

            def read(self, *args, **kwargs):
                return pd.DataFrame()

        with pytest.raises(TypeError):
            IncompleteIO(data_dir=Path("/tmp"))

    def test_read_method_can_be_called(self, tmp_path):
        """Test that read method can be called with flexible signature."""
        io = TestBaseIO(data_dir=tmp_path)
        result = io.read()
        assert isinstance(result, pd.DataFrame)

    def test_read_method_with_args_kwargs(self, tmp_path):
        """Test that read method accepts arbitrary args and kwargs."""
        io = TestBaseIO(data_dir=tmp_path)
        # Should not raise any exception
        result = io.read(1, 2, 3, key1="value1", key2="value2")
        assert isinstance(result, pd.DataFrame)

    def test_download_and_process_method_can_be_called(self, tmp_path):
        """Test that download_and_process method can be called with flexible signature."""
        io = TestBaseIO(data_dir=tmp_path)
        # Should not raise any exception
        io.download_and_process()

    def test_download_and_process_method_with_args_kwargs(self, tmp_path):
        """Test that download_and_process accepts arbitrary args and kwargs."""
        io = TestBaseIO(data_dir=tmp_path)
        # Should not raise any exception
        io.download_and_process(1, 2, 3, key1="value1", key2="value2")


class TestBaseIOAttributes:
    """Test BaseIO class and instance attributes."""

    def test_env_var_name_attribute(self):
        """Test that ENV_VAR_NAME class attribute is set correctly."""
        assert TestBaseIO.ENV_VAR_NAME == "TEST_DATA_DIR"

    def test_label_attribute(self):
        """Test that LABEL class attribute is set correctly."""
        assert TestBaseIO.LABEL == "test_label"

    def test_data_dir_instance_attribute(self, tmp_path):
        """Test that data_dir is an instance attribute."""
        io = TestBaseIO(data_dir=tmp_path)
        assert hasattr(io, "data_dir")
        assert isinstance(io.data_dir, Path)

    def test_data_dir_is_path_instance(self, tmp_path):
        """Test that data_dir is converted to Path instance."""
        io = TestBaseIO(data_dir=str(tmp_path))
        assert isinstance(io.data_dir, Path)

    def test_multiple_instances_have_independent_data_dirs(self, tmp_path):
        """Test that multiple instances can have different data_dirs."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"

        io1 = TestBaseIO(data_dir=dir1)
        io2 = TestBaseIO(data_dir=dir2)

        assert io1.data_dir == dir1
        assert io2.data_dir == dir2
        assert io1.data_dir != io2.data_dir


class TestBaseIOEnvironmentVariablePrecedence:
    """Test environment variable precedence logic."""

    def test_env_var_not_set_uses_passed_dir(self, tmp_path):
        """Test that passed data_dir is used when env var is not set."""
        passed_dir = tmp_path / "passed"

        with patch.dict(os.environ, {}, clear=True):
            io = TestBaseIO(data_dir=passed_dir)
            assert io.data_dir == passed_dir

    def test_env_var_set_and_data_dir_passed_uses_data_dir(self, tmp_path):
        """Test that passed data_dir takes precedence when prefer_env_var is False."""
        env_dir = tmp_path / "env"
        passed_dir = tmp_path / "passed"

        with patch.dict(os.environ, {"TEST_DATA_DIR": str(env_dir)}):
            io = TestBaseIO(data_dir=passed_dir, prefer_env_var=False)
            assert io.data_dir == passed_dir

    def test_prefer_env_var_true_overrides_data_dir(self, tmp_path):
        """Test that env var takes precedence when prefer_env_var is True."""
        env_dir = tmp_path / "env"
        passed_dir = tmp_path / "passed"

        with patch.dict(os.environ, {"TEST_DATA_DIR": str(env_dir)}):
            io = TestBaseIO(data_dir=passed_dir, prefer_env_var=True)
            assert io.data_dir == env_dir


class TestBaseIOUrl:
    """Test the `url` property."""

    def test_url_returns_single_url(self, tmp_path):
        """Test that url property returns the subclass's URL string as-is."""

        class SingleUrl(TestBaseIO):
            URL = "https://example.com/data/file.txt"

        io = SingleUrl(data_dir=tmp_path)
        assert io.url == "https://example.com/data/file.txt"

    def test_url_returns_list_of_urls(self, tmp_path):
        """Test that url property returns a list as-is for multi-file sources."""

        class MultiUrl(TestBaseIO):
            URL = ["https://example.com/a.txt", "https://example.com/b.txt"]

        io = MultiUrl(data_dir=tmp_path)
        assert io.url == ["https://example.com/a.txt", "https://example.com/b.txt"]

    def test_url_can_override_fallback(self, tmp_path):
        """Test that subclasses can override _fallback_url when the base attribute isn't named URL."""

        class ComputedUrl(TestBaseIO):
            API_URL = "https://example.com/file.txt"

            def _fallback_url(self):
                return self.API_URL

        io = ComputedUrl(data_dir=tmp_path)
        assert io.url == "https://example.com/file.txt"

    def test_url_reflects_last_resolved_request(self, tmp_path):
        """Test that url returns the fully resolved URL after a request was recorded."""

        class RecordingUrl(TestBaseIO):
            URL = "https://example.com/data/"

        io = RecordingUrl(data_dir=tmp_path)
        assert io.url == "https://example.com/data/"

        io._record_url("https://example.com/data/2026-07-29.txt")
        assert io.url == "https://example.com/data/2026-07-29.txt"

    def test_url_returns_list_when_multiple_requests_recorded(self, tmp_path):
        """Test that url returns a list when more than one request was made."""

        class MultiRequestUrl(TestBaseIO):
            URL = "https://example.com/data/"

        io = MultiRequestUrl(data_dir=tmp_path)
        io._record_url("https://example.com/data/a.txt")
        io._record_url("https://example.com/data/b.txt")
        assert io.url == ["https://example.com/data/a.txt", "https://example.com/data/b.txt"]
