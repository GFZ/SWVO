# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod


class BaseReader:
    def __init__(self):
        self.data_folder = None

    @abstractmethod
    def read(self, *args):
        raise NotImplementedError("Read method for Base Reader not implemented.")

    @abstractmethod
    def _check_data_folder(self):
        raise NotImplementedError("Check data folder method for Base Reader not implemented.")
