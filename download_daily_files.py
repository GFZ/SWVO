# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

# flake8: ignore=E722

import os
import logging
import argparse

from datetime import datetime, timedelta, timezone
from pathlib import Path
import traceback
from urllib.error import HTTPError

from swvo.io.dst import DSTWDC
from swvo.io.kp import KpNiemegk, KpSWPC
from swvo.io.hp import Hp30GFZ, Hp60GFZ
from swvo.io.omni import OMNILowRes, OMNIHighRes
from swvo.io.solar_wind import SWACE, DSCOVR
from swvo.io.f10_7 import F107SWPC, F107OMNI



def get_parser():
    parser = argparse.ArgumentParser(description="Download and process daily external data files.")
    parser.add_argument('--logs', action="store", default=None, type=str,
                        help="Absolute path to the log folder")
    return parser

def main(args):

    time_now = datetime.now(timezone.utc)

    log_dir = Path(args.logs) / f"{time_now.year}" / f"{time_now.month}"

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

    logging.info("OMNI low resolution...\n")
    try:
        OMNILowRes().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logging.error("Encountered error while downloading OMNI low res. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("OMNI high resolution...\n")
    try:
        OMNIHighRes().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
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
        logging.error("Encountered error while downloading DSCOVR solar wind. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("DST WDC...")
    try:
        DSTWDC().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logging.error("Encountered error while downloading WDC Dst data. Traceback:")
        logging.error(traceback.format_exc())

    logging.info("Finished downloading and processing!")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)