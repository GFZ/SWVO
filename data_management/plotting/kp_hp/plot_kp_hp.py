import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import pandas as pd
import numpy as np
import logging
from data_management.plotting.plotting_base import PlotOutput

matplotlib.use('Agg')


class PlotKpHpOutput(PlotOutput):
    """
    This class is in charge to produce standard plots for Kp/Hp output data.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _add_bar_color(data, key):
        color = []
        for i in range(len(data)):
            if data[key][i] <= 3:
                color.append([0.0, 0.5, 0.0, 1.0])
            elif (3 < data[key][i] and data[key][i] <= 6):
                color.append([204 / 255.0, 204 / 255.0, 0.0, 1.0])
            elif 6 < data[key][i]:
                color.append([1.0, 0.0, 0.0, 1.0])
        return color

    @staticmethod
    def _get_max_ylim(data, data_column):
        if data_column == "kp":
            return 9.1
        elif data_column in ["hp60", "hp30"]:
            return np.maximum(9.1, np.nanmax(data[data_column].values) + 0.1)
        if data_column.startswith("kp") or data_column.startswith("Kp"):
            return 9.1
        elif data_column.startswith("hp") or data_column.startswith("Hp"):
            return np.maximum(9.1, np.nanmax(data[data_column].values) + 0.1)
        

    @staticmethod
    def _set_yaxis_style(ax, ylim, ylabel_fontsize=20, ylabel=None):
        ax.set_ylim(ylim)
        y_labels = [i for i in range(int(ylim[1]) + 1)]
        ax.set_yticks(y_labels)
        ax.tick_params(axis="y", labelsize=ylabel_fontsize, direction='in')
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, rotation=90, labelpad=15)
        return ax

    @staticmethod
    def _add_subplot(ax, data, data_column, title=None, rotation=0, title_font=9, xlabel_fontsize=14,
                    width=0.9, align="edge", alpha=None, bar_colors=None):

        if bar_colors is None:
            bar_colors = PlotKpHpOutput._add_bar_color(data, list(data.keys())[0])
        ax = data[data_column].plot(kind="bar", ax=ax, edgecolor=['k'] * len(data), color=bar_colors,
                                    align=align, width=width, legend=False, alpha=alpha)
        ax.set_title(title, fontsize=title_font)

        first_hour = data.index[0].hour

        def map_dates(x):

            if ((x.hour - first_hour) % 6 != 0 ) or (x.minute != 0):
                return ""
            elif ((x.hour - first_hour) % 6 == 0) and (x.hour == first_hour):
                return x.strftime("%H:%M\n%d %b")
            else:
                return x.strftime("%H:%M")

        ax.set_xlabel("Time (UTC)", fontsize=15, labelpad=10)

        x_labels = list(data.index.map(lambda x: map_dates(x)))
        ax.set_xticklabels(labels=x_labels, rotation=rotation, fontsize=xlabel_fontsize)

        ax.grid(True, axis='y', linestyle='dashed')

        return ax

    @staticmethod
    def _check_column_to_plot(data, column_to_plot):
        if column_to_plot not in data.columns:
            raise ValueError("specified data column {} is not among the data"
                             " columns {}".format(column_to_plot,
                                                  data.columns))
    @staticmethod
    def _get_label(column_to_plot):
        if column_to_plot == "kp":
            return "K_{p}"
        elif column_to_plot == "hp60":
            return "H_{p}60"
        elif column_to_plot == "hp30":
            return "H_{p}30"
        else:
            raise ValueError("given value for {} not expected".format(column_to_plot))


    def plot_output(self, data, column_to_plot, ax=None, legend=True):
        """
        This function plots output data for Kp/Hp products. The plot format is at the moment fixed.

        :param data: This is the standard output format of Kp/Hp products read by KpReader class.
        :type data: pandas.DataFrame
        :param column_to_plot: column in data to plot.
        :type column_to_plot: str
        :param ax: An Axes object in the case the plot needs to be combined with other figures outside of the class
                   otherwise pass None.
        :type ax: matplotlib.axes.Axes or None
        :param legend: If True the default legend of the plot is kept, otherwise it is not plotted.
        :type legend: bool
        :return: An Axes object
        """

        if not isinstance(data, pd.DataFrame):
            msg = ("data must be an instance of a pandas dataframe, instead "
                   "it is of type {}").format(type(data))
            logging.error(msg)
            raise TypeError(msg)

        PlotKpHpOutput._check_column_to_plot(data, column_to_plot)

        label = PlotKpHpOutput._get_label(column_to_plot)

        if ax is None:
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(1, 1, 1)

        ax = PlotKpHpOutput._add_subplot(ax, data=data[[column_to_plot]],
                                         data_column=column_to_plot,
                                         title=None)

        max_ylim = PlotKpHpOutput._get_max_ylim(data, column_to_plot)
        ax = PlotKpHpOutput._set_yaxis_style(ax, ylim=(-0.1, max_ylim),
                                             ylabel_fontsize=20,
                                             ylabel=r'${}$'.format(label))

        red_patch = patches.Patch(color='red', label=r'${}$ > 6'.format(label))
        yellow_patch = patches.Patch(color=[204 / 255.0, 204 / 255.0, 0.0, 1.0],
                                     label=r'3 < ${}$ <= 6'.format(label))
        green_patch = patches.Patch(color='green', label=r'${}$ <= 3'.format(label))
        transparent_patch = patches.Patch(color=[0, 0, 0, 0.1],
                                          label='Data not available')

        if legend:
            ax.legend(bbox_to_anchor=(0., 1., 0.84, .275),
                      handles=[green_patch, yellow_patch, red_patch,
                               transparent_patch],
                      ncol=4, fontsize="x-large", shadow=True)

        if ax is None:
            fig.subplots_adjust(left=None, bottom=0.3, right=None, top=0.7,
                                wspace=None, hspace=0.6)

        return ax


class PlotKpHpEnsembleOutput(PlotKpHpOutput):
    """
    This class is in charge to produce standard plots for Kp/Hp output data.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _add_max_bars(ax, data, color='r', max_bar=None):
        if max_bar is None:
            for i, d in enumerate(data.index):
                ax.hlines(y=data.values[i][0], xmin=i + 0.03, xmax=i + 0.9, linewidth=4, color=color)
        else:
            PlotKpHpEnsembleOutput._check_max_bar_keys(max_bar)
            for i, d in enumerate(data.index):
                ax.hlines(y=data.values[i][0],
                          xmin=i + max_bar["xmin_shift"],
                          xmax=i + max_bar["max_shift"],
                          linewidth=max_bar["linewidht"], color=color)
        return ax

    @staticmethod
    def _check_max_bar_keys(max_bar):
        supported_parameters = ["xmin_shift", "max_shift", "linewidht"]
        if not (set(max_bar.keys()) == set(supported_parameters)):
            raise ValueError("not all the necessary keys of provided max_bar a"
                             "re provided")

    def plot_output(self, data, column_to_plot, legend=True, max_bar=None):
        """
        This function plots output data for Kp/Hp Ensemble. The plot format is at the moment
        fixed.

        :param data: This is the standard output format of Kp Ensemble products read by KpEnsembleReader class.
        :type data: list of pandas.DataFrame
        :param column_to_plot: column in data to plot.
        :type column_to_plot: str
        :param legend: If True the default legend of the plot is kept, otherwise it is not plotted.
        :type legend: bool
        :return: An Axes object
        """

        if not isinstance(data, list):
            msg = "Data argument passed must be an instance of a list, instead it is of type {}".format(type(data))
            logging.error(msg)
            raise TypeError(msg)

        for d in data:
            if not isinstance(d, pd.DataFrame):
                msg = "Each element of data must be an instance of a pandas dataframe, instead" \
                      " it is of type {}".format(type(d))
                logging.error(msg)
                raise TypeError(msg)

            PlotKpHpEnsembleOutput._check_column_to_plot(d, column_to_plot)

        label = PlotKpHpEnsembleOutput._get_label(column_to_plot)

        data_median = pd.DataFrame(np.median([d[[column_to_plot]].values.flatten() for d in data], axis=0),
                                   columns=[column_to_plot],
                                   index=data[0].index)
        data_max = pd.DataFrame(np.max([d[[column_to_plot]].values.flatten() for d in data], axis=0),
                                columns=[column_to_plot], index=data[0].index)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax = PlotKpHpEnsembleOutput._add_subplot(ax, data_median,
                                                 column_to_plot,
                                                 title=None)
        ax = PlotKpHpEnsembleOutput._add_max_bars(ax,
                                                  data=data_max,
                                                  max_bar=max_bar)

        max_ylim = PlotKpHpEnsembleOutput._get_max_ylim(data_max,
                                                        column_to_plot)
        PlotKpHpEnsembleOutput._set_yaxis_style(ax, ylim=(-0.1, max_ylim),
                                                ylabel_fontsize=20,
                                                ylabel=r'${}$'.format(label))

        red_patch = patches.Patch(color='red', label=r'${}$ > 6'.format(label))
        yellow_patch = patches.Patch(color=[204 / 255.0, 204 / 255.0, 0.0, 1.0],
                                     label=r'3 < ${}$ <= 6'.format(label))
        green_patch = patches.Patch(color='green', label=r'${}$ <= 3'.format(label))
        transparent_patch = patches.Patch(color=[0, 0, 0, 0.1], label='Data not available')

        if legend:
            ax.legend(bbox_to_anchor=(0., 1., 0.84, .275),
                      handles=[green_patch, yellow_patch, red_patch, transparent_patch],
                      ncol=4, fontsize="x-large", shadow=True)

        if ax is None:
            fig.subplots_adjust(left=None, bottom=0.3, right=None, top=0.7, wspace=None, hspace=0.6)

        return ax
