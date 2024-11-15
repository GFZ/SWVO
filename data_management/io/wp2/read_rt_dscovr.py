import datetime as dt
import glob
import json
import logging
import os

import numpy as np
import pandas as pd

from data_management.io.base_file_reader import BaseReader


class DSCOVRRTReader(BaseReader):
    """
    This class reads
    """
    DATA_FIELDS = ["proton_density", "speed", "bx", "by", "bz", "b", "temperature"]

    def __init__(self,
                 dscovr_output_folder="/PAGER/WP6/data/outputs/"
                                      "RBM_Forecast/realtime_stream/dscovr_realtime_stream/json/"):
        """
        :param dscovr_output_folder: The path to the output folder of WP2 products.
        :type dscovr_output_folder: str
        """
        super().__init__()
        self.data_folder = dscovr_output_folder
        self._check_data_folder()

    def _check_data_folder(self):
        if not os.path.exists(self.data_folder):
            msg = "Data folder {} for WP2 DSCOVR REAL-TIME solar wind output not found...impossible to retrieve data.".format(
                self.data_folder)
            logging.error(msg)
            raise FileNotFoundError(msg)

    @staticmethod
    def _to_date(x):
        year = int(x["year"])
        month = int(x["month"])
        day = int(x["day"])
        hour = int(str(x["time"])[0:2])
        minute = int(str(x["time"])[2:4])
        return dt.datetime(year, month, day, hour, minute)

    @staticmethod
    def _read_mag_file(filename):
        data_mag = json.load(open(filename))[1:]
        header_mag = ["t", "bx_gsm", "by_gsm", "bz_gsm", "lon", "lat", "bavg"]
        data_mag = pd.DataFrame(np.array(data_mag), columns=header_mag)
        data_mag["t"] = pd.to_datetime(data_mag["t"])
        for k in header_mag[1:]:
            data_mag[k] = data_mag[k].astype(np.float32)
        data_mag.index = data_mag["t"]
        data_mag.drop(["t", "lat", "lon"], axis=1, inplace=True)
        return data_mag

    @staticmethod
    def _read_plasma_file(filename):
        header_sw = ["t", "proton_density", "speed", "temperature"]
        data_sw = json.load(open(filename))[1:]
        data_sw = pd.DataFrame(np.array(data_sw), columns=header_sw)
        data_sw["t"] = pd.to_datetime(data_sw["t"])
        for k in header_sw[1:]:
            data_sw[k] = data_sw[k].astype(np.float32)
        data_sw.index = data_sw["t"]
        data_sw.drop(["t"], axis=1, inplace=True)
        return data_sw

    def read(self, date=None, fields=None) -> pd.DataFrame:
        """
        This function reads output data from SWIFT and returns it in the form of a tuple of two pandas dataframe,
        each for each coordinate system available, GSM and HGC.

        :param date: The date in which data has been produced. It assumes that the data is produced once a day. If None
                     the data with current date is requested.
        :type date: datetime.datetime or None
        :param fields: List of fields to be extracted from the available data.
        :type fields: list or None
        :raises: FileNotFoundError: when one of the data files requested is not found.
                 KeyError: when a field requested is not among available field list
        :return: Tuple of GSM and HGC data as pandas data frames
        """
        if date is None:
            date = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        if fields is not None:
            for f in fields:
                if f not in DSCOVRRTReader.DATA_FIELDS:
                    msg = "Requested field from DSCOVR Real time data not available..."
                    logging.error(msg)
                    raise KeyError(msg)

        files_plasma = glob.glob(os.path.join(self.data_folder, "plasma-1-day-{}{}{}{}*.json".format(str(date.year),
                                                                                         str(date.month).zfill(2),
                                                                                         str(date.day).zfill(2),
                                                                                         str(date.hour).zfill(2),
                                                                                         str(date.minute).zfill(2))))
        files_mag = glob.glob(os.path.join(self.data_folder, "mag-1-day-{}{}{}{}*.json".format(str(date.year),
                                                                                   str(date.month).zfill(2),
                                                                                   str(date.day).zfill(2),
                                                                                   str(date.hour).zfill(2),
                                                                                   str(date.minute).zfill(2))))

        data_plasma = DSCOVRRTReader._read_plasma_file(sorted(files_plasma)[0])
        data_mag = DSCOVRRTReader._read_mag_file(sorted(files_mag)[0])
        data = pd.concat([data_plasma, data_mag], axis=1)
        if fields is not None:
            data = data[fields]
        return data
