import datetime as dt
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class SWSWIFTEnsemble:
    PROTON_MASS = 1.67262192369e-27

    ENV_VAR_NAME = "SWIFT_ENSEMBLE_OUTPUT_DIR"
    LABEL = "swift"

    def __init__(self, data_dir: str | Path = None):
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)

        logging.info(f"SWIFT ensemble data directory: {self.data_dir}")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist! Impossible to retrieve data!")

    def read(self, start_time: datetime, end_time: datetime) -> list:
        if start_time and not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start_time is None:
            start_time = datetime.now(timezone.utc).replace(microsecond=0, minute=0, second=0)

        if end_time is None:
            end_time = start_time.replace(tzinfo=timezone.utc) + timedelta(days=3)

        str_date = start_time.strftime("%Y%m%dt0000")

        ensemble_folders = list((self.data_dir / str_date).glob("*task*"))

        logging.info(f"Found {len(ensemble_folders)} SWIFT tasks folders...")
        gsm_s = []

        for ensemble_folder in ensemble_folders:
            try:
                file = list((ensemble_folder / "SWIFT").glob("gsm_*"))[0]
                data_gsm = self._read_single_file(file)
                data_gsm = data_gsm.truncate(
                    before=start_time - timedelta(minutes=10),
                    after=end_time + timedelta(minutes=10),
                )

                gsm_s.append(data_gsm)
            except IndexError:
                msg = f"GSM SWIFT output file for date {str_date} and task {ensemble_folder} not found...impossible to read"
                logging.warning(msg)

        return gsm_s

    def read_single_output(self, target_time: datetime):
        pass

    def _read_single_file(self, file_name, use_old_column_names=False) -> pd.DataFrame:
        """
        This function reads one the two available json file of SWIFT output and extract relevant variables
        combining them into a pandas DataFrame.

        The path to the file to read.
        :param file_name: The path of the file to read.
        :type file_name: str
        :param fields: Lists of fields to extract from the DataFrame. The list needs to contain a subset
                       of available fields. if None, all the fields available are retrieved.
        :type fields: list
        :return: A pandas.DataFrame with requested variables.
        """
        with open(file_name) as f:
            data = json.load(f)

        time = list(
            map(
                lambda x: dt.datetime.fromtimestamp(int(x), tz=dt.timezone.utc),
                data["arrays"]["Unix time"]["data"],
            )
        )

        ux = np.array(data["arrays"]["Vx"]["data"]) / 1000.0
        uy = np.array(data["arrays"]["Vy"]["data"]) / 1000.0
        uz = np.array(data["arrays"]["Vz"]["data"]) / 1000.0

        bx = np.array(data["arrays"]["Bx"]["data"]) * 1.0e9
        by = np.array(data["arrays"]["By"]["data"]) * 1.0e9
        bz = np.array(data["arrays"]["Bz"]["data"]) * 1.0e9

        temperature = np.array(data["arrays"]["Temperature_ion"]["data"])

        speed = np.sqrt(ux**2 + uy**2 + uz**2)
        b = np.sqrt(bx**2 + by**2 + bz**2)

        n = np.array(data["arrays"]["Rho"]["data"]) / self.PROTON_MASS * 1.0e-6

        if use_old_column_names:
            df = pd.DataFrame(
                {
                    "proton_density": n,
                    "speed": speed,
                    "b": b,
                    "temperature": temperature,
                    "bx": bx,
                    "by": by,
                    "bz": bz,
                    "ux": ux,
                    "uy": uy,
                    "uz": uz,
                },
                index=time,
            )
        else:
            df = pd.DataFrame(
                {
                    "proton_density": n,
                    "speed": speed,
                    "bavg": b,
                    "temperature": temperature,
                    "bx_gsm": bx,
                    "by_gsm": by,
                    "bz_gsm": bz,
                },
                index=time,
            )

        df["file_name"] = file_name

        return df
