from abc import ABC

from data_management.check.base_file_checker import BaseFileCheck
from data_management.notifications.email_notifier import send_failure_email

import datetime as dt
import os
import glob
import logging
from email.headerregistry import Address


class RBMForecastCheck(BaseFileCheck, ABC):
    def __init__(self, wp6_data_folder="/PAGER/WP6/data/outputs/",
                 product_sub_folder="/RBM_Forecast/rbm_forecast/fullMat/*/Archive/"):
        super().__init__()
        self.wp_folder = wp6_data_folder
        self.product_sub_folder = product_sub_folder
        self.subject_email = "PAGER WP6, RBM Forecast Module, DATA FAILURE..."
        self.email_recipients = self._get_email_recipients()

    @staticmethod
    def _get_email_recipients():
        addresses = tuple()
        addresses += (Address("Ruggero Vasile", "ruggero.vasile", "gfz-potsdam.de"),
                      Address("Ingo Michaelis", "ingo.michaelis", "gfz-potsdam.de"))
        return addresses

    @staticmethod
    def _extract_date_from_file(file):
        file = file.split("/")[-1]
        file = file.split(".")[0]
        date = file.split("_")[-2]
        time = file.split("_")[-1]
        timestamp = dt.datetime.strptime(date + time, "%Y%m%d%H%M%S")
        return timestamp

    def check_files_exists(self, check_date):

        time_to_check = check_date.replace(minute=0, second=0, microsecond=0)
        file_list = glob.glob(os.path.join(self.wp_folder, self.product_sub_folder) + "/*")
        for file in file_list:
            date = RBMForecastCheck._extract_date_from_file(file)
            if date == time_to_check:
                return True
        return False

    def run_check(self, date=None, notify=False):
        """
        It runs a check about existence of given outputs of rbm forecast module. If the files are not found
        for a given date provided, it sends an email to a default list of users and notifies them
        of the problem. If date is not specified then the check is performed on the date of the day
        the script has been called.

        We will add also a check on the file format and content later on...

        :param date: Date on which to run the check.
        :type date: datetime.datetime
        :param notify: True if you want to use email notifications, otherwise False
        :type notify: bool
        """
        success = self.check_files_exists(date)
        if not success:
            logging.warning("RBM Forecast outputs for date {} not found...".format(date))
            if notify:
                logging.warning("Sending notification email")
                content = "Output files not generated yet today..."
                send_failure_email(subject=self.subject_email, content=content, addresses_to=self.email_recipients,
                                   address_from=self.email_sender)
        else:
            logging.info("RBM Forecast outputs for date {} found!!".format(date))
