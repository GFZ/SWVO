import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple

import numpy as np
import pandas as pd
import wget


class OMNILowRes:
    ENV_VAR_NAME = "OMNI_LOW_RES_STREAM_DIR"

    URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"

    HEADER = [
        "year",
        "day",
        "hour",
        "BarRotNumber",
        "id_imf",
        "id_sw",
        "%points_imfavg",
        "%points_plasmaavg",
        "B_mag_avg",
        "bavg",
        "lat_angle_avg_field",
        "lon_angle_avg_field",
        "bx_gse_gsm",
        "by_gse",
        "bz_gse",
        "by_gsm",
        "bz_gsm",
        "sigma_mod_B",
        "sigma_B",
        "sigma_Bx",
        "sigma_By",
        "sigma_Bz",
        "temperature",
        "proton_density",
        "speed",
        "speed_angle_lon",
        "speed_angle_lat",
        "alpha_proton_ratio",
        "flow_pressure",
        "sigma_T",
        "sigma_N",
        "sigma_V",
        "sigma_phi_V",
        "sigma_theta_V",
        "sigma_alpha_proton_ratio",
        "e",
        "plasma_beta",
        "alfven_mach_n",
        "Kp",
        "sunspot_n",
        "dst",
        "ae",
        "p_flux_1",
        "p_flux_2",
        "p_flux_4",
        "p_flux_10",
        "p_flux_30",
        "p_flux_60",
        "flag",
        "ap",
        "f107",
        "pc",
        "al",
        "au",
        "magnetosonic_mach_n",
    ]

    def __init__(self, data_dir: str | Path = None):
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"OMNI Low Res  data directory: {self.data_dir}")

    def download_and_process(
        self, start_time: datetime, end_time: datetime, reprocess_files: bool = False, verbose: bool = False
    ):
        temporary_dir = Path("./temp_omni_low_res_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)

            for file_path, time_interval in zip(file_paths, time_intervals):
                filename = "omni2_" + str(time_interval[0].year) + ".dat"

                if file_path.exists():
                    if reprocess_files:
                        file_path.unlink()
                    else:
                        continue

                if verbose:
                    logging.info(f"Downloading file {self.URL + filename} ...")

                wget.download(self.URL + filename, str(temporary_dir))

                if verbose:
                    logging.info(f"Processing file ...")

                processed_df = self._process_single_file(temporary_dir / filename)
                processed_df.to_csv(file_path, index=True, header=True)

        finally:
            rmtree(temporary_dir)

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> Tuple[List, List]:
        file_paths = []
        time_intervals = []

        current_time = datetime(start_time.year, 1, 1, 0, 0, 0)
        end_time = datetime(end_time.year, 12, 31, 23, 59, 59)

        while current_time < end_time:
            file_path = self.data_dir / f"OMNI_LOW_RES_{current_time.strftime('%Y')}.csv"
            file_paths.append(file_path)

            interval_start = current_time
            interval_end = datetime(current_time.year, 12, 31, 23, 59, 59)

            time_intervals.append((interval_start, interval_end))
            current_time = datetime(current_time.year + 1, 1, 1, 0, 0, 0)

        return file_paths, time_intervals

    def _process_single_file(self, file_path):
        data = pd.read_csv(file_path, sep=r"\s+", names=self.HEADER)
        data["timestamp"] = data["year"].map(str).apply(lambda x: x + "-01-01 ")
        data["timestamp"] = data["year"].map(str).apply(lambda x: x + "-01-01 ") + data["hour"].map(str).apply(
            lambda x: x.zfill(2)
        )
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["timestamp"] = data["timestamp"] + data["day"].apply(lambda x: timedelta(days=int(x) - 1))
        data.set_index("timestamp", inplace=True)
        mask = data["dst"] >= 99998
        data.loc[mask, "dst"] = np.nan

        mask = data["Kp"] == 99
        data.loc[mask, "Kp"] = np.nan

        df = pd.DataFrame(index=data.index)
        df["dst"] = data["dst"]
        df["kp"] = data["Kp"] / 10
        df["f107"] = data["f107"]

        # change rounded numbers to be equal to 1/3 or 2/3 to be consistent with other Kp products
        df.loc[round(df["kp"] % 1, 2) == 0.7, "kp"] = round(df.loc[round(df["kp"] % 1, 2) == 0.7, "kp"]) - 1 / 3
        df.loc[round(df["kp"] % 1, 2) == 0.3, "kp"] = round(df.loc[round(df["kp"] % 1, 2) == 0.3, "kp"]) + 1 / 3

        return df

    def read(self, start_time: datetime, end_time: datetime, download: bool = False) -> pd.DataFrame:
        START_YEAR = 1963

        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)

        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start_time < datetime(START_YEAR, 1, 1).replace(tzinfo=timezone.utc):
            logging.warning(
                "Start date chosen falls behind the existing data. Moving start date to first"
                " available mission files..."
            )
            start_time = datetime(START_YEAR, 1, 1, tzinfo=timezone.utc)

        assert start_time < end_time

        file_paths, _ = self._get_processed_file_list(start_time, end_time)

        dfs = []

        for file_path in file_paths:
            if not file_path.exists():
                if download:
                    self.download_and_process(start_time, end_time)
                else:
                    logging.warning(f"File {file_path} not found")
                    continue

            dfs.append(self._read_single_file(file_path))

        data_out = pd.concat(dfs)

        if not data_out.empty:
            if not data_out.index.tzinfo:
                data_out.index = data_out.index.tz_localize("UTC")

        return data_out

    def _read_single_file(self, file_path) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        df["t"] = pd.to_datetime(df["timestamp"])
        df.index = df["t"]

        df["file_name"] = file_path
        df.loc[df["kp"].isna(), "file_name"] = None

        return df
