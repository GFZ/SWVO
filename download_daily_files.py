# flake8: ignore=E722

import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import traceback

from data_management.io.kp import KpNiemegk, KpSWPC
from data_management.io.hp import Hp30GFZ, Hp60GFZ
from data_management.io.omni import OMNILowRes, OMNIHighRes
from data_management.io.solar_wind import SWACE
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


# ENV_VAR_NAMES = {"DAILY_DOWNLOAD_LOG_DIR": "/home/pager/FLAG_DEV/logs/daily_downloads_external_data/"}
ENV_VAR_NAMES = {"DAILY_DOWNLOAD_LOG_DIR": "/home/pager/FLAG_DEV/code/external_data/data_management/data/logs"}

if __name__ == "__main__":

    for key, var in ENV_VAR_NAMES.items():
        os.environ[key] = ENV_VAR_NAMES[key]

    time_now = datetime.now(timezone.utc)

    log_dir = Path(os.environ.get("DAILY_DOWNLOAD_LOG_DIR")) / f"{time_now.year}" / f"{time_now.month}"

    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(log_dir / f'daily_downloads_log_{time_now.strftime("%Y%m%d_T%H0000")}.log'),
        filemode="w",
        format="%(asctime)s,%(msecs)d %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    date_yesterday_start = time_now.replace(hour=0, minute=0, second=1) - timedelta(days=1)
    date_yesterday_end = date_yesterday_start.replace(hour=23, minute=59, second=59)

    logging.info(f"Target time from {date_yesterday_start} to {time_now}")

    logging.info("Starting downloading and processing...")

    logging.info("Kp Niemegk...\n")
    try:
        KpNiemegk().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except:
        logging.error("Encountered error while downloading Niemegk Kp. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("Kp SWPC...\n")
    try:
        KpSWPC().download_and_process(time_now, reprocess_files=True)
    except:
        logging.error("Encountered error while downloading SWPC Kp. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("OMNI low resolution...\n")
    try:
        OMNILowRes().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except:
        logging.error("Encountered error while downloading OMNI low res. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("OMNI high resolution...\n")
    try:
        OMNIHighRes().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except:
        logging.error("Encountered error while downloading OMNI high res. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("Hp30 GFZ...\n")
    try:
        Hp30GFZ().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except:
        logging.error("Encountered error while downloading Hp30. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("Hp60 GFZ...\n")
    try:
        Hp60GFZ().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except:
        logging.error("Encountered error while downloading Hp30. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("SW ACE RT...\n")
    try:
        SWACE().download_and_process(time_now)
    except:
        logging.error("Encountered error while downloading ACE RT solar wind. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("F10.7 SWPC RT...\n")
    try:
        F107SWPC().download_and_process()
    except:
        logging.error("Encountered error while downloading F10.7cm solar wind. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("F10.7 OMNI low resolution...\n")
    try:
        F107OMNI().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except:
        logging.error("Encountered error while downloading F10.7cm solar wind. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("Finished downloading and processing!")
