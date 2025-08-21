# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module handling SW data from OMNI High Resolution files.
"""


from swvo.io.omni import OMNIHighRes

class SWOMNI(OMNIHighRes):
    """
    Class for reading SW data from OMNI High resolution files. 
    Inherits the :func:`download_and_process`, other private methods and attributes from :class:`OMNIHighRes`.
    """
    def __init__(self, data_dir: str = None):
        """
        Initialize a SWOMNI object.

        Parameters
        ----------
        data_dir : str | None, optional
            Data directory for the OMNI SW data. If not provided, it will be read from the environment variable
        """
        super().__init__(data_dir=data_dir)
