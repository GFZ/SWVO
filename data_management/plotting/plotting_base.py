# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod


class PlotOutput(object):
    def __init__(self):
        self.description = None

    @abstractmethod
    def plot_output(self, *args):
        raise NotImplementedError
