from data_management.io.omni import OMNIHighRes

class SWOMNI(OMNIHighRes):
    """
    Class for reading SW data from OMNI High resolution files. 
    Inherits the `download_and_process`, other private methods and attributes from OMNIHighRes.
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
