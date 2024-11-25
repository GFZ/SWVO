import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple
import logging

import numpy as np
import pandas as pd
import wget


class OMNIHighRes(object):

    ENV_VAR_NAME = "OMNI_HIGH_RES_STREAM_DIR"

    URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/"

    START_YEAR = 1981

    HEADER = [
        "year",
        "day",
        "hour",
        "minute",
        "id_imf",
        "id_sw",
        "%points_imfavg",
        "%points_plasmaavg",
        "%interp",
        "timeshift",
        "rms_timeshift",
        "rms_phase_front",
        "time_btwn_observation",
        "bavg",
        "bx_gse_gsm",
        "by_gse",
        "bz_gse",
        "by_gsm",
        "bz_gsm",
        "rms_sd_scalar",
        "rms_sd_vector",
        "speed",
        "vx_gse",
        "vy_gse",
        "vz_gse",
        "proton_density",
        "temperature",
        "flow_pressure",
        "e",
        "plasma_beta",
        "alfven_mach_n",
        "x_gse",
        "y_gse",
        "z_gse",
        "bsn_x",
        "bsn_y",
        "bsn_z",
        "ae",
        "al",
        "au",
        "sym_d",
        "sym_h",
        "asy_d",
        "asy_h",
        "pc",
        "magnetosonic_mach_n",
        "p_flux_10",
        "p_flux_30",
        "p_flux_60",
    ]

    def __init__(self, data_dir: str | Path = None):

        if data_dir is None:

            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"OMNI high resolution data directory: {self.data_dir}")

    def download_and_process(
        self,
        start_time: datetime,
        end_time: datetime,
        cadence_min: float = 1,
        reprocess_files: bool = False,
        verbose: bool = False,
    ):

        assert cadence_min == 1 or cadence_min == 5, "Only 1 or 5 minute cadence can be chosen for high resolution omni data."

        temporary_dir = Path("./temp_omni_high_res_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            file_paths, time_intervals = self._get_processed_file_list(start_time, end_time, cadence_min)

            for file_path, time_interval in zip(file_paths, time_intervals):

                if cadence_min == 1:
                    filename = "omni_min" + str(time_interval[0].year) + ".asc"
                elif cadence_min == 5:
                    filename = "omni_5min" + str(time_interval[0].year) + ".asc"

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

    def read(self, start_time: datetime, end_time: datetime, cadence_min: float = 1, download: bool = False) -> pd.DataFrame:

        assert cadence_min == 1 or cadence_min == 5, "Only 1 or 5 minute cadence can be chosen for high resolution omni data."

        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)

        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start_time < datetime(self.START_YEAR, 1, 1, tzinfo=timezone.utc):
            logging.warning("Start date chosen falls behind the existing data. Moving start date to first" " available mission files...")
            start_time = datetime(self.START_YEAR, 1, 1, tzinfo=timezone.utc)

        assert start_time < end_time

        file_paths, _ = self._get_processed_file_list(start_time, end_time, cadence_min)

        dfs = []

        for file_path in file_paths:

            if not file_path.exists():
                if download:
                    self.download_and_process(start_time, end_time, cadence_min=cadence_min)
                else:
                    logging.warning(f"File {file_path} not found")
                    continue

            dfs.append(self._read_single_file(file_path))

        data_out = pd.concat(dfs)

        if not data_out.empty:
            if not data_out.index.tzinfo:
                data_out.index = data_out.index.tz_localize("UTC")

        data_out = data_out.truncate(
            before=start_time - timedelta(minutes=cadence_min - 0.0000001),
            after=end_time + timedelta(minutes=cadence_min + 0.0000001),
        )

        return data_out

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime, cadence_min: float) -> Tuple[List, List]:

        file_paths = []
        time_intervals = []

        current_time = datetime(start_time.year, 1, 1, 0, 0, 0)
        end_time = datetime(end_time.year, 12, 31, 23, 59, 59)

        while current_time < end_time:

            file_path = self.data_dir / f"OMNI_HIGH_RES_{cadence_min}min_{current_time.strftime('%Y')}.csv"
            file_paths.append(file_path)

            interval_start = current_time
            interval_end = datetime(current_time.year, 12, 31, 23, 59, 59)

            time_intervals.append((interval_start, interval_end))
            current_time = datetime(current_time.year + 1, 1, 1, 0, 0, 0)

        return file_paths, time_intervals

    def _process_single_file(self, file_path):

        to_drop = [
            "year",
            "day",
            "hour",
            "minute",
            "id_imf",
            "id_sw",
            "%points_imfavg",
            "%points_plasmaavg",
            "%interp",
            "timeshift",
            "rms_timeshift",
            "rms_phase_front",
            "time_btwn_observation",
            "rms_sd_scalar",
            "rms_sd_vector",
            "plasma_beta",
            "alfven_mach_n",
            "bsn_x",
            "bsn_y",
            "bsn_z",
            "ae",
            "al",
            "au",
            "sym_d",
            "sym_h",
            "asy_d",
            "asy_h",
            "pc",
            "magnetosonic_mach_n",
            "p_flux_10",
            "p_flux_30",
            "p_flux_60",
            "e",
            "flow_pressure",
            "by_gse",
            "bz_gse",
            "vx_gse",
            "vy_gse",
            "vz_gse",
            "x_gse",
            "y_gse",
            "z_gse",
            "bx_gse",
        ]

        maxes = {
            "bavg": 9999.9,
            "bx_gse": 9999.9,
            "bx_gsm": 9999.9,
            "by_gse": 9999.9,
            "bz_gse": 9999.9,
            "by_gsm": 9999.9,
            "bz_gsm": 9999.9,
            "speed": 99999.8,
            "vx_gse": 99999.8,
            "vy_gse": 99999.8,
            "vz_gse": 99999.8,
            "proton_density": 999.8,
            "temperature": 9999998.0,
            "x_gse": 9999.9,
            "y_gse": 9999.9,
            "z_gse": 9999.9,
        }

        data = pd.read_csv(file_path, sep=r"\s+", names=self.HEADER)

        data["timestamp"] = data["year"].map(str).apply(lambda x: x + "-01-01 ") + data["hour"].map(str).apply(
            lambda x: x.zfill(2)
        )
        data["timestamp"] += data["minute"].map(str).apply(lambda x: ":" + x.zfill(2) + ":00")
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["timestamp"] = data["timestamp"] + data["day"].apply(lambda x: timedelta(days=int(x) - 1))
        data["bx_gse"] = data["bx_gse_gsm"]
        data["bx_gsm"] = data["bx_gse_gsm"]

        data.drop(to_drop + ["bx_gse_gsm"], axis=1, inplace=True)
        data.set_index("timestamp", inplace=True)

        for k in data:
            mask = data[k] > maxes[k]
            data.loc[mask, k] = np.nan

        return data

    def _read_single_file(self, file_path) -> pd.DataFrame:

        df = pd.read_csv(file_path)

        df["t"] = pd.to_datetime(df["timestamp"], utc=True)
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)
        df.drop(labels=["timestamp"], axis=1, inplace=True)

        nan_mask = df.isna().all(axis=1)
        df["file_name"] = file_path
        df.loc[nan_mask, "file_name"] = None

        return df
