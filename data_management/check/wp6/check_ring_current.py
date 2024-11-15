from abc import ABC

from data_management.check.base_file_checker import BaseFileCheck
from data_management.notifications.email_notifier import send_failure_email

import datetime as dt
import os
import glob
import logging
from email.headerregistry import Address


class RingCurrentCheck(BaseFileCheck, ABC):
    RC_FILE_TEMPLATE_NAME = "VERB4D_RC_Forecast"

    def __init__(self, wp6_data_folder, product_sub_folder):
        super().__init__()
        self.wp_folder = wp6_data_folder
        self.product_sub_folder = product_sub_folder
        self.subject_email = "PAGER WP6, Ring Current Module, DATA FAILURE..."
        self.email_recipients = self._get_email_recipients()

    @staticmethod
    def _get_email_recipients():
        addresses = tuple()
        addresses += (Address("Ruggero Vasile", "ruggero.vasile", "gfz-potsdam.de"),
                      Address("Michael Wutzig", "michael.wutzig", "gfz-potsdam.de"))
        return addresses

    @staticmethod
    def _extract_date_from_file(file):
        file = file.split("/")[-1]
        timestamp = file.split("_")[-1]
        timestamp = dt.datetime.strptime(timestamp, "%Y%m%dT%H%M")
        return timestamp

    def check_files_exists(self, check_date):
        time_to_check = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
        file_list = sorted(glob.glob(
            os.path.join(self.wp_folder, self.product_sub_folder) + "/{}*".format(self.RC_FILE_TEMPLATE_NAME)))
        for file in file_list:
            date = self._extract_date_from_file(file)
            if date == time_to_check:
                return True
        return False

    def run_check(self, date, notify=False):
        """
        It runs a check about existence of given outputs of ring current module. If the files are not found
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
            logging.warning("Ring current outputs for date {} not found...".format(date.date()))
            if notify:
                logging.warning("Sending notification email")
                content = "Output files not generated yet today..."
                send_failure_email(subject=self.subject_email, content=content, addresses_to=self.email_recipients,
                                   address_from=self.email_sender)
        else:
            logging.info("Ring Current outputs for date {} found!!".format(date.date()))

