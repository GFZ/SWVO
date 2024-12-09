from data_management.io.omni import OMNIHighRes


class SWOMNI(OMNIHighRes):
    def __init__(self, data_dir: str = None):
        super().__init__(data_dir=data_dir)
