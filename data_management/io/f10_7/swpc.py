from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple
import warnings


import pandas as pd
import numpy as np
import wget
from data_management.io.decorators import (
    add_time_docs,
    add_attributes_to_class_docstring,
    add_methods_to_class_docstring,
)

logging.captureWarnings(True)


@add_attributes_to_class_docstring
@add_methods_to_class_docstring
class F107SWPC:
    """This is a class for the SWPC F107 data.

    Parameters
    ----------
    data_dir : str | Path, optional
        Data directory for the OMNI Low Resolution data. If not provided, it will be read from the environment variable

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "RT_SWPC_F107_DIR"
    URL = "https://services.swpc.noaa.gov/text/"
    NAME_F107 = "daily-solar-indices.txt"

    LABEL = "swpc"

    def __init__(self, data_dir: str | Path | None = None):
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                msg = f"Necessary environment variable {self.ENV_VAR_NAME} not set!"
                raise ValueError(msg)
            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"SWPC F10.7 data directory: {self.data_dir}")

    @add_time_docs(None)
    def _get_processed_file_list(
        self, start_time: datetime, end_time: datetime
    ) -> tuple[list[Path], list[tuple[datetime, datetime]]]:
        """Get list of file paths and their corresponding time intervals.

        Returns
        -------
        Tuple[List, List]
            List of file paths and time intervals.
        """
        years_needed = range(start_time.year, end_time.year + 1)

        file_paths = [self.data_dir / f"SWPC_F107_{year}.csv" for year in years_needed]
        time_intervals = [
            (
                datetime(year, 1, 1, tzinfo=timezone.utc),
                datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
            for year in years_needed
        ]

        return file_paths, time_intervals

    def download_and_process(self) -> None:
        """Download and process the latest 30-day F10.7 data.

        Returns
        -------
        None
        """
        temp_dir = Path("./temp_f107")
        temp_dir.mkdir(exist_ok=True)

        try:
            logging.debug("Downloading F10.7 data...")

            wget.download(self.URL + self.NAME_F107, str(temp_dir))

            if Path(temp_dir / self.NAME_F107).stat().st_size == 0:
                msg = f"Error downloading file: {self.URL + self.NAME_F107}"
                raise FileNotFoundError(msg)

            logging.debug("Processing F10.7 data...")

            new_data = self._process_single_file(temp_dir / self.NAME_F107)

            for year, year_data in new_data.groupby(new_data.date.dt.year):
                file_path = self.data_dir / f"SWPC_F107_{year}.csv"

                if file_path.expanduser().exists():
                    logging.debug(f"Updating {file_path}...")

                    existing_data = pd.read_csv(file_path, parse_dates=["date"])
                    existing_data["date"] = pd.to_datetime(
                        existing_data["date"]
                    ).dt.tz_localize(None)

                    combined_data = pd.concat([existing_data, year_data])
                    combined_data = combined_data.drop_duplicates(
                        subset=["date"], keep="last"
                    )
                    combined_data = combined_data.sort_values("date")

                    new_records = len(combined_data) - len(existing_data)
                    logging.debug(f"Added {new_records} new records to {year}")
                else:
                    logging.debug(f"Creating new file for {year}")
                    combined_data = year_data

                combined_data.to_csv(file_path, index=False)

        finally:
            # ...
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_single_file(self, file_path: Path) -> pd.DataFrame:
        """Read and process the F10.7 data file.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from yearly F10.7 file.
        """

        data = pd.read_csv(
            file_path,
            sep=r"\s+",
            skiprows=13,
            usecols=[0, 1, 2, 3],
            names=["year", "month", "day", "f107"],
        )

        data["date"] = pd.to_datetime(data[["year", "month", "day"]].assign(hour=0))
        data = data[["date", "f107"]]

        return data  # noqa: RET504

    @add_time_docs("read")
    def read(
        self, start_time: datetime, end_time: datetime, *, download: bool = False
    ) -> pd.DataFrame:
        """Read F10.7 SWPC data for the given time range.

        Parameters
        ----------
        download : bool, optional
            Download data on the go, defaults to False.

        Returns
        -------
        pd.DataFrame
            F10.7 data.

        Raises
        ------
        ValueError
            Raises ValueError if start_time is after end_time."
        """

        if start_time >= end_time:
            msg = "start_time must be before end_time"
            raise ValueError(msg)

        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        file_paths, _ = self._get_processed_file_list(start_time, end_time)
        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(
                end_time.year,
                end_time.month,
                end_time.day,
            ),
            freq=timedelta(days=1),
        )
        data_out = pd.DataFrame(index=t)
        data_out["f107"] = np.array([np.nan] * len(t))
        data_out["date"] = data_out.index
        data_out["file_name"] = np.array([None] * len(t))

        for file_path in file_paths:
            if not file_path.exists():
                if download:
                    self.download_and_process()
                else:
                    warnings.warn(f"File {file_path} not found")
                    continue

            df_one_file = self._read_single_file(file_path)
            data_out = df_one_file.combine_first(data_out)

        if not data_out.empty:
            if data_out.index.tzinfo is None:
                data_out.index = data_out.index.tz_localize("UTC")
        data_out.drop("date", axis=1, inplace=True)
        data_out = data_out.truncate(
            before=start_time - timedelta(hours=2.9999),
            after=end_time + timedelta(hours=2.9999),
        )

        return data_out

    def _read_single_file(self, file_path: Path) -> pd.DataFrame:
        """Read yearly F107 file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from yearly F10.7 SWPC Resolution file.
        """
        df = pd.read_csv(file_path)

        df["date"] = pd.to_datetime(df["date"])
        df.index = df["date"]

        df["file_name"] = file_path
        df.loc[df["f107"].isna(), "file_name"] = None

        return df
