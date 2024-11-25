import os
import shutil
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple
import logging
import pandas as pd
import wget


class F107SWPC:
    ENV_VAR_NAME = "RT_SWPC_F107_DIR"
    URL = "https://services.swpc.noaa.gov/text/"
    NAME_F107 = "daily-solar-indices.txt"

    def __init__(self, data_dir: str | Path = None):

        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")
            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"SWPC F10.7 data directory: {self.data_dir}")

    def _get_processed_file_list(
        self, start_time: datetime, end_time: datetime
    ) -> Tuple[List[Path], List[Tuple[datetime, datetime]]]:
        """Returns list of file paths and their corresponding time intervals."""
        years_needed = range(start_time.year, end_time.year + 1)

        file_paths = [self.data_dir / f"SWPC_F107_{year}.csv" for year in years_needed]
        time_intervals = [(datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)) for year in years_needed]

        return file_paths, time_intervals

    def download_and_process(self, verbose: bool = False) -> pd.DataFrame:
        """Downloads and processes the latest 30-day F10.7 data."""
        temp_dir = Path("./temp_f107")
        temp_dir.mkdir(exist_ok=True)

        try:
            if verbose:
                logging.info("Downloading F10.7 data...")

            wget.download(self.URL + self.NAME_F107, str(temp_dir))

            if os.stat(temp_dir / self.NAME_F107).st_size == 0:
                raise FileNotFoundError(f"Error downloading file: {self.URL + self.NAME_F107}")

            if verbose:
                logging.info("Processing F10.7 data...")

            new_data = self._read_f107_file(temp_dir / self.NAME_F107)

            for year, year_data in new_data.groupby(new_data.date.dt.year):
                file_path = self.data_dir / f"SWPC_F107_{year}.csv"

                if file_path.expanduser().exists():
                    if verbose:
                        logging.info(f"Updating {file_path}...")

                    existing_data = pd.read_csv(file_path, parse_dates=["date"])
                    existing_data["date"] = pd.to_datetime(existing_data["date"]).dt.tz_localize(None)

                    combined_data = pd.concat([existing_data, year_data])
                    combined_data = combined_data.drop_duplicates(subset=["date"], keep="last")
                    combined_data = combined_data.sort_values("date")

                    if verbose:
                        new_records = len(combined_data) - len(existing_data)
                        logging.info(f"Added {new_records} new records to {year}")
                else:
                    if verbose:
                        logging.info(f"Creating new file for {year}")
                    combined_data = year_data

                combined_data.to_csv(file_path, index=False)

        finally:
            # ...
            shutil.rmtree(temp_dir)

    def _read_f107_file(self, file_path: Path) -> pd.DataFrame:
        """Reads and processes the F10.7 data file."""
        data = pd.read_csv(file_path, sep=r"\s+", skiprows=13, usecols=[0, 1, 2, 3], names=["year", "month", "day", "f107"])

        data["date"] = pd.to_datetime(data[["year", "month", "day"]].assign(hour=0))
        data = data[["date", "f107"]]
        return data

    def read(self, start_time: datetime, end_time: datetime, download: bool = False) -> pd.DataFrame:

        assert start_time < end_time, "start_time must be before end_time"

        current_year = datetime.now().year
        file_paths, file_intervals = self._get_processed_file_list(start_time, end_time)

        years_requested = range(start_time.year, end_time.year + 1)

        if download:
            self.download_and_process()

        available_years = [path.stem[-4:] for path in self.data_dir.glob("SWPC_F107_*.csv")]

        missing_files = [year for year in years_requested if str(year) not in available_years and year < current_year]

        if missing_files:
            logging.warning(f"Data for year(s) {', '.join(map(str, missing_files))} not found.")

            if len(available_years) != 0:
                logging.warning(f"Only data for {', '.join(available_years)} are available.")
            else:
                logging.warning("No data available. Set `download` to `True`")

        dfs = []

        available_file_paths = []
        for path in file_paths:
            if any(year in str(path) for year in available_years):
                available_file_paths.append(path)
            else:
                logging.warning(f"File {path} not found")

        for file_path in available_file_paths:
            year_data = pd.read_csv(file_path, parse_dates=["date"])
            dfs.append(year_data)

        if not dfs:
            return pd.DataFrame(columns=["date", "f107"])

        data_out = pd.concat(dfs)
        return data_out
