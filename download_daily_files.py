# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

# flake8: ignore=E722

import logging
import os
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from swvo.io.dst import DSTWDC
from swvo.io.f10_7 import F107SWPC
from swvo.io.hp import Hp30GFZ, Hp60GFZ
from swvo.io.kp import KpNiemegk, KpSWPC
from swvo.io.omni import OMNIHighRes, OMNILowRes
from swvo.io.solar_wind import SWACE, SWIMAP, SWMIDL
from swvo.logger import setup_logging

# Any of these that isn't already set in the environment falls back to "./data", so the script
# can run out of the box without every source's directory being configured beforehand.
REQUIRED_ENV_VARS = {
    DSTWDC.ENV_VAR_NAME,
    F107SWPC.ENV_VAR_NAME,
    Hp30GFZ.ENV_VAR_NAME,
    Hp60GFZ.ENV_VAR_NAME,
    KpNiemegk.ENV_VAR_NAME,
    KpSWPC.ENV_VAR_NAME,
    OMNIHighRes.ENV_VAR_NAME,
    OMNILowRes.ENV_VAR_NAME,
    SWACE.ENV_VAR_NAME,
    SWIMAP.ENV_VAR_NAME,
    SWMIDL.ENV_VAR_NAME,
}

for _env_var in REQUIRED_ENV_VARS:
    if _env_var not in os.environ:
        os.environ[_env_var] = "./data"

LOGS_DIR = os.environ.get("DAILY_DOWNLOAD_LOGS_DIR", "./logs")

time_now = datetime.now(timezone.utc)

log_dir = Path(LOGS_DIR) / f"{time_now.year}" / f"{time_now.month:02d}"

log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"daily_downloads_log_{time_now.strftime('%Y%m%d_T%H0000')}.log"

setup_logging(log_file=log_file)


logger = logging.getLogger(__name__)


def main():
    date_yesterday_start = time_now.replace(hour=0, minute=0, second=1) - timedelta(days=1)
    # date_yesterday_end = date_yesterday_start.replace(hour=23, minute=59, second=59)

    logger.info(f"Target time from {date_yesterday_start} to {time_now}")

    logger.info("Starting downloading and processing...")

    logger.info("Kp Niemegk...")
    try:
        KpNiemegk().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logger.error("Encountered error while downloading Niemegk Kp. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("Kp SWPC...")
    try:
        KpSWPC().download_and_process(time_now, reprocess_files=True)
    except Exception:
        logger.error("Encountered error while downloading SWPC Kp. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("OMNI low resolution...\n")
    try:
        OMNILowRes().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logger.error("Encountered error while downloading OMNI low res. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("OMNI high resolution...\n")
    try:
        OMNIHighRes().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logger.error("Encountered error while downloading OMNI high res. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("Hp30 GFZ...")
    try:
        Hp30GFZ().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logger.error("Encountered error while downloading Hp30. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("Hp60 GFZ...")
    try:
        Hp60GFZ().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logger.error("Encountered error while downloading Hp30. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("SW ACE RT...")
    try:
        SWACE().download_and_process(date_yesterday_start, time_now)
    except Exception:
        logger.error("Encountered error while downloading ACE RT solar wind. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("F10.7 SWPC RT...")
    try:
        F107SWPC().download_and_process()
    except Exception:
        logger.error("Encountered error while downloading F10.7cm solar wind. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("SW MIDL...")
    try:
        SWMIDL().download_and_process(date_yesterday_start, time_now)
    except Exception:
        logger.error("Encountered error while downloading MIDL solar wind. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("SW IMAP...")
    try:
        SWIMAP().download_and_process(date_yesterday_start, time_now)
    except Exception:
        logger.error("Encountered error while downloading IMAP solar wind. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("DST WDC...")
    try:
        DSTWDC().download_and_process(date_yesterday_start, time_now, reprocess_files=True)
    except Exception:
        logger.error("Encountered error while downloading WDC Dst data. Traceback:")
        logger.error(traceback.format_exc())

    logger.info("Finished downloading and processing!")


if __name__ == "__main__":
    main()
