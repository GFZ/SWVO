from abc import abstractmethod


class PlotOutput(object):
    def __init__(self):
        self.description = None

    @abstractmethod
    def plot_output(self, *args):
        raise NotImplementedError
