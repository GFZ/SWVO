import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from data_management.plotting.plotting_base import PlotOutput

matplotlib.use('Agg')


class PlotSWIFTOutput(PlotOutput):
    """
    This class is in charge to produce standard plots for SWIFT WP2 output data.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _add_subplot(ax, data, title=None, title_font=9, ylabel_fontsize=10, ylabel=None,
                     line_width=1, color='orange', legend=False, label="SWIFT"):

        ax = data.plot(ax=ax, legend=legend, linewidth=line_width, color=color, label=label)
        ax.set_title(title, fontsize=title_font)

        # Y-AXIS
        ax.tick_params(axis="y", labelsize=ylabel_fontsize, direction='in')
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, rotation=90, labelpad=0)

        # X-AXIS
        ax.tick_params(axis="x", which="both", bottom=True)

        # GRID
        ax.grid(True, axis='y', linestyle='dashed')

        return ax


    @staticmethod
    def plot_ensemble_output(data, color="orange", legend=False, label="SWIFT", linewidth=1):
        """
        This function plots output data for SWIFT ensemble solar wind variables.

        :param data: This is the standard output format of SWIFT Ensemble products read by SWIFTEnsembleReader class.
        :type data: list of pandas.DataFrame
        :return: A figure object of type matplotlib.figure.Figure containing the produced plot
        """

        fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 10))

        fig.supxlabel("Time (UTC)", fontsize=15)

        locator = matplotlib.dates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = matplotlib.dates.ConciseDateFormatter(locator)
        ax[3].xaxis.set_major_locator(locator)
        ax[3].xaxis.set_major_formatter(formatter)

        for d in data:

            PlotSWIFTOutput._add_subplot(ax[0], d["speed"], ylabel=r"$|U|(km/s)$",
                                         color=color, legend=legend,
                                         label=label,
                                         line_width=linewidth)
            PlotSWIFTOutput._add_subplot(ax[1], d["proton_density"],
                                         ylabel=r"$N_{p}(cm^{-3})$",
                                         color=color, legend=legend,
                                         label=label,
                                         line_width=linewidth)
            PlotSWIFTOutput._add_subplot(ax[2], d["bavg"], ylabel=r"$|B|(nT)$",
                                         color=color,
                                         legend=legend, label=label,
                                         line_width=linewidth)
            PlotSWIFTOutput._add_subplot(ax[3], d["bz_gsm"],
                                         ylabel=r"$Bz(nT)$",
                                         color=color, legend=legend,
                                         label=label)

        return fig, ax
