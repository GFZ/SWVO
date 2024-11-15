from abc import ABC

from data_management.check.base_file_checker import BaseFileCheck
from data_management.notifications.email_notifier import send_failure_email

import os
import deprecation
import glob
import logging
from email.headerregistry import Address


class KpDataCheck(BaseFileCheck, ABC):
    SWPC_FILE_TEMPLATE_NAME = "SWPC_KP_FORECAST"
    NIEMEGK_FILE_TEMPLATE_NAME = "NIEMEGK_KP_NOWCAST"
    L1_FORECAST_FILE_TEMPLATE_NAME = "FORECAST"

    def __init__(self, wp3_data_folder, product_sub_folder):
        super().__init__()
        self.wp_folder = wp3_data_folder
        self.product_sub_folder = product_sub_folder
        self.subject_email = "PAGER WP3, KP Module, DATA FAILURE..."
        self.email_recipients = self._get_email_recipients()

    @staticmethod
    def _get_email_recipients():
        addresses = tuple()
        addresses += (Address("Ruggero Vasile", "ruggero", "gfz-potsdam.de"),)
        return addresses

    def _check_swpc_file_exists(self, check_date):

        check_date = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
        check_date_str = check_date.strftime("%Y%m%d")
        try:
            path = os.path.join(self.wp_folder, self.product_sub_folder,
                                self.SWPC_FILE_TEMPLATE_NAME + "_{}.csv".format(check_date_str))
            file = sorted(glob.glob(path))[0]
            success = True
            logging.info("SWPC output Kp for date {} found!!".format(check_date.date()))
        except IndexError:
            file = None
            success = False
            logging.warning("SWPC KP forecast for date {} not found ...".format(check_date.date()))
        return success, file

    def _check_niemegk_file_exists(self, check_date):

        check_date = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
        check_date_str = check_date.strftime("%Y%m%d")
        try:
            path = os.path.join(self.wp_folder, self.product_sub_folder,
                                self.NIEMEGK_FILE_TEMPLATE_NAME + "_{}.csv".format(check_date_str))
            file = sorted(glob.glob(path))[0]
            success = True
            logging.info("NIEMEGK output Kp for date {} found!!".format(check_date.date()))
        except IndexError:
            file = None
            success = False
            logging.warning("NIEMEGK KP nowcast for date {} not found ...".format(check_date.date()))
        return success, file

    @deprecation.deprecated("This function is deprecated and will be substituted by the swift kp ensemble check")
    def _check_swift_file_exists(self, check_date):

        check_date = check_date.replace(minute=0, second=0, microsecond=0)
        check_date_str = check_date.strftime("%Y%m%dT%H%M%S")
        try:
            file = glob.glob(self.wp_folder + "/SWIFT/FORECAST_PAGER_SWIFT_swift_{}.csv".format(check_date_str))[0]
            success = True
            logging.info("SWIFT output Kp for date {} found!!".format(check_date))
        except IndexError:
            file = None
            success = False
            logging.warning("SWIFT Kp forecast for date {} not found ...".format(check_date))
        return success, file

    def _check_l1_file_exists(self, model, spc, check_date):

        check_date = check_date.replace(minute=0, second=0, microsecond=0)
        check_date_str = check_date.strftime("%Y%m%dT%H%M%S")
        try:
            path = os.path.join(self.wp_folder, self.product_sub_folder,
                                self.L1_FORECAST_FILE_TEMPLATE_NAME + "_{}_{}_{}.csv".format(model,
                                                                                             spc, check_date_str))
            file = sorted(glob.glob(path))[0]
            success = True
            logging.info("L1 Kp forecast for date {},"
                         " spacecraft source {} and model {} found!!".format(check_date, spc, model))
        except IndexError:
            file = None
            success = False
            logging.warning("L1 Kp forecast for date {},"
                            " spacecraft source {} and model {} not found ...".format(check_date, spc, model))
        return success, file

    def run_check(self, product, model=None, date=None, notify=False):
        """
        It runs a check about existence of given outputs of WP3 Kp modules. If the files are not found
        for a given date provided, it sends an email to a default list of users and notifies them
        of the problem. If date is not specified then the check is performed on the date of the day
        the script has been called.

        We will add also a check on the file format and content later on...

        :param product: The Kp product to check, choose among "l1", "swpc", "niemegk"
        :type product: str
        :param model: Used at the moment for product L1. It is the name of the model to check for outputs.
        :type model: str
        :param date: Date on which to run the check.
        :type date: datetime.datetime
        :param notify: True if you want to use email notifications, otherwise False
        :type notify: bool
        """
        if product == "swpc":
            success, file, = self._check_swpc_file_exists(date)
        elif product == "niemegk":
            success, file, = self._check_niemegk_file_exists(date)
        elif product == "l1":
            if model is None:
                msg = "Kp product checker for l1 output needs a model name to proceed"
                logging.error(msg)
                raise RuntimeError(msg)
            success, file, = self._check_l1_file_exists(model, "dscovr_rt", date)
        else:
            msg = "Kp product checker product {} not among listed outputs..."
            logging.error(msg)
            raise RuntimeError(msg)

        if not success:
            content = "Output files for Kp WP3 module {} not generated yet today...".format(product)
            logging.warning(content)
            if notify:
                logging.warning("Sending email notification...")
                send_failure_email(subject=self.subject_email, content=content, addresses_to=self.email_recipients,
                                   address_from=self.email_sender)
