# flake8: ignore=E722

import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import traceback
from urllib.error import HTTPError

from data_management.io.kp import KpNiemegk, KpSWPC
from data_management.io.hp import Hp30GFZ, Hp60GFZ
from data_management.io.omni import OMNILowRes, OMNIHighRes
from data_management.io.solar_wind import SWACE, DSCOVR
from data_management.io.f10_7 import F107SWPC, F107OMNI

"""
base_path = '/home/bhaas/FLAG_TEST/'
kp_path = base_path + 'Kp/'

ENV_VAR_NAMES = {'DAILY_DOWNLOAD_LOG_DIR': base_path + 'log/',
                 'RT_KP_NIEMEGK_STREAM_DIR': kp_path + 'Niemegk/',
                 'RT_KP_SWPC_STREAM_DIR': kp_path + 'SWPC/',
                 'OMNI_LOW_RES_STREAM_DIR': base_path + 'OMNI/Low_res',
                 'RT_HP_GFZ_STREAM_DIR' : base_path + 'HP/'}
"""


ENV_VAR_NAMES = {"DAILY_DOWNLOAD_LOG_DIR": "/home/pager/FLAG_DEV/logs/daily_downloads_external_data/"}
# ENV_VAR_NAMES = {"DAILY_DOWNLOAD_LOG_DIR": "/home/pager/FLAG_DEV/code/external_data/data_management/data/logs"}

if __name__ == "__main__":

    for key, var in ENV_VAR_NAMES.items():
        os.environ[key] = ENV_VAR_NAMES[key]

    time_now = datetime.now(timezone.utc)

    log_dir = Path(os.environ.get("DAILY_DOWNLOAD_LOG_DIR")) / f"{time_now.year}" / f"{time_now.month}"

    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(log_dir / f'daily_downloads_log_{time_now.strftime("%Y%m%d_T%H0000")}.log'),
        filemode="w",
        format="%(asctime)s,%(msecs)d - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    date_yesterday_start = time_now.replace(hour=0, minute=0, second=1) - timedelta(days=1)
    date_yesterday_end = date_yesterday_start.replace(hour=23, minute=59, second=59)

    logging.info(f"Target time from {date_yesterday_start} to {time_now}")

    logging.info("Starting downloading and processing...")

    logging.info("Kp Niemegk...")
    try:
        KpNiemegk().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logging.error("Encountered error while downloading Niemegk Kp. Traceback:")
        logging.error(traceback.format_exc())


    logging.info("Kp SWPC...")
    try:
        KpSWPC().download_and_process(time_now, reprocess_files=True)
    except Exception:
        logging.error("Encountered error while downloading SWPC Kp. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("OMNI low resolution...")
    omni_low_res_start_date = date_yesterday_start
    omni_low_res_end_date = time_now
    while True:
        try:
            OMNILowRes().download_and_process(omni_low_res_start_date, omni_low_res_end_date, reprocess_files=True)
            break

        except HTTPError as e:
            logging.error(f"HTTPError: {e}")
            logging.error(f"Could not download file for {omni_low_res_start_date.year}.")
            logging.info(f"Reverting to {date_yesterday_start.year - 1}")
            omni_low_res_start_date = omni_low_res_start_date.replace(year=omni_low_res_start_date.year - 1)
            omni_low_res_end_date = datetime(omni_low_res_start_date.year, 12, 31)
            continue

        except Exception:
            logging.error("Encountered error while downloading OMNI low res. Traceback:")
            logging.error(traceback.format_exc())


    logging.info("OMNI high resolution...")
    omni_high_res_start_date = date_yesterday_start
    omni_high_res_end_date = time_now
    while True:
        try:
            OMNIHighRes().download_and_process(date_yesterday_start, omni_high_res_end_date, reprocess_files=True)
            break

        except HTTPError as e:
            logging.error(f"HTTPError: {e}")
            logging.error(f"Could not download file for {omni_high_res_start_date.year}.")
            logging.info(f"Reverting to {date_yesterday_start.year - 1}")
            omni_high_res_start_date = omni_high_res_start_date.replace(year=omni_high_res_start_date.year - 1)
            omni_high_res_end_date = datetime(omni_high_res_start_date.year, 12, 31)
            continue

        except Exception:
            logging.error("Encountered error while downloading OMNI high res. Traceback:")
            logging.error(traceback.format_exc())

    logging.info("Hp30 GFZ...")
    try:
        Hp30GFZ().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logging.error("Encountered error while downloading Hp30. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("Hp60 GFZ...")
    try:
        Hp60GFZ().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logging.error("Encountered error while downloading Hp30. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("SW ACE RT...")
    try:
        SWACE().download_and_process(time_now)
    except Exception:
        logging.error("Encountered error while downloading ACE RT solar wind. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("F10.7 SWPC RT...")
    try:
        F107SWPC().download_and_process()
    except Exception:
        logging.error("Encountered error while downloading F10.7cm solar wind. Traceback:")
        logging.error(traceback.format_exc())
    logging.info("SW DSCOVR...")
    try:
        DSCOVR().download_and_process(time_now)
    except Exception:
        logging.error("Encountered error while downloading DSCOVR data. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("Finished downloading and processing!")
