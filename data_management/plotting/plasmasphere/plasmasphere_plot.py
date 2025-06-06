import os

import glob
import logging

import subprocess

import math

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from data_management.plotting.plotting_base import PlotOutput


class PlasmaspherePlot(PlotOutput):

    def __init__(self, path_c_map=None):

        super().__init__()

        self.figure = None
        self.ax = None
        if path_c_map is None:
            base_path = os.path.dirname(__file__)
            self.path_c_map = os.path.abspath(os.path.join(base_path, 'my_cmap.npy'))
        else:
            self.path_c_map = path_c_map

        self.colour_map_density = mpl.colors.ListedColormap(np.load(self.path_c_map))
        self.colour_map_deviation = "OrRd"

    @staticmethod
    def _get_date_components(date):
        """
        It gets a datetime instance and returns year, month, day, hour, minute

        :param date: a date
        :type date: an instance of datetime object
        :return: year, month, day, hour, minute
        :rtype: tuple of int
        """
        year = str(date.year)
        month = date.strftime('%m')
        day = date.strftime('%d')
        hour = date.strftime('%H')
        minute = date.strftime("%M")
        return year, month, day, hour, minute

    @staticmethod
    def _draw_earth_night_side(axes):
        axes.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 100),
                          0, 1, alpha=1, color='white')
        return axes

    @staticmethod
    def _set_axes_ticks_labels(axes):

        ticks_loc = [x for x in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        axes.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        axes.set_xticklabels(
            labels=[xlabel for xlabel in np.arange(0, 24, 3)]
        )

        ticks_loc = [1, 2, 3, 4, 5, 6]
        axes.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        axes.set_ylim(0, 6.5)
        axes.set_yticklabels(
            labels=ticks_loc
        )
        axes.tick_params(axis="y", colors="black")
        axes.grid(c='grey', lw=0.5, ls='-', visible=True)
        return axes

    @staticmethod
    def _plot_polar_distribution(axes,
                                 angle_values,
                                 l_values,
                                 distribution_values,
                                 colour_map,
                                 title, vmin, vmax):

        unique_l_values = np.unique(l_values)
        unique_angle_values = np.unique(angle_values)

        angle_grid = np.reshape(angle_values,
                                (len(unique_l_values), len(unique_angle_values)),
                                order="F")
        l_grid = np.reshape(l_values,
                            (len(unique_l_values), len(unique_angle_values)),
                            order="F")

        distribution_on_grid = np.reshape(
            distribution_values,
            (len(unique_l_values), len(unique_angle_values)),
            order="F")
        axes.contourf(angle_grid, l_grid, distribution_on_grid,
                      levels=512, cmap=colour_map, vmin=vmin, vmax=vmax)

        axes = PlasmaspherePlot._draw_earth_night_side(axes)
        axes = PlasmaspherePlot._set_axes_ticks_labels(axes)
        axes.set_title(title, fontdict={'fontweight': 'bold'})

        return axes

    @staticmethod
    def _add_colour_bar(axes, colour_map,
                        min_value, max_value, label):

        colour_bar_normalization = mpl.colors.Normalize(vmin=min_value, vmax=max_value)

        color_bar = plt.colorbar(mpl.cm.ScalarMappable(
            norm=colour_bar_normalization,
            cmap=colour_map
        ),
            ax=axes,
            use_gridspec=True,
            shrink=0.5,
            pad=0.15
        )
        color_bar.set_label(label)
        return axes

    @staticmethod
    def _mlt_to_angle(mlt):
        return mlt * 2 * math.pi / 24

    def _set_date(self, date):
        if not isinstance(date, datetime):
            msg = "date must be an instance of a datetime object"
            logging.error(msg)
            raise ValueError(msg)
        else:
            self.date = date

    def _set_figure(self, fig):
        self.figure = fig

    @staticmethod
    def _predictions_all_nan(density_values):
        if density_values.isnull().values.all():
            return True

    @staticmethod
    def _get_mean(two_d_array):
        return np.nanmean(two_d_array, axis=1)

    @staticmethod
    def _get_max(two_d_array):
        return np.nanmax(two_d_array, axis=1)

    @staticmethod
    def _get_min(two_d_array):
        return np.nanmin(two_d_array, axis=1)

    @staticmethod
    def _get_deviation(density_values):
        return np.nanstd(density_values, axis=1)

    @staticmethod
    def _plot_time_series_statistics(ax, df, time_column, target_columns):
        mean_kp = PlasmaspherePlot._get_mean(df[target_columns].values)
        ax.plot(df[time_column], mean_kp)
        max_value = PlasmaspherePlot._get_max(df[target_columns].values)
        min_value = PlasmaspherePlot._get_min(df[target_columns].values)
        ax.fill_between(df[time_column], max_value, min_value, alpha=.5)
        return ax

    @staticmethod
    def _plot_input(ax, df, time_column, input_columns):

        if len(input_columns) == 1:
            ax.plot(df[time_column], df[input_columns[0]])

        elif len(input_columns) > 1:
            ax = PlasmaspherePlot._plot_time_series_statistics(
                ax, df, time_column=time_column,
                target_columns=input_columns
            )
        else:
            raise RuntimeError("input_columns list has lenght 0")

        return ax


    def _plot_single_date_prediction(self, l_values, mlt_values,
                                     density_values, date,
                                     fig_size=(16, 9), df_kp=None,
                                     df_solar_wind=None):

        self._set_date(date)
        fig = plt.figure(figsize=fig_size)
        self.figure = fig
        grid = fig.add_gridspec(4, 3)

        angle_values = PlasmaspherePlot._mlt_to_angle(mlt_values)
        if len(list(density_values.columns)) == 1:
            ax = self.figure.add_subplot(grid[:, 0],
                                         projection='polar')
            self.ax = ax
            self.ax = PlasmaspherePlot._plot_polar_distribution(
                self.ax,
                angle_values,
                l_values,
                density_values.values,
                colour_map=self.colour_map_density,
                title="{} UTC".format(self.date.strftime("%Y-%m-%d, %H:%M")),
                vmin=0,
                vmax=3.99
            )
            self.ax = PlasmaspherePlot._add_colour_bar(
                self.ax,
                self.colour_map_density,
                min_value=0, max_value=3.99,
                label='$log_{10}(n_e)$')
        else:
            ax_mean = self.figure.add_subplot(grid[0:2, 0],
                                              projection='polar')
            self.ax_mean = ax_mean
            mean_densities = PlasmaspherePlot._get_mean(density_values.values)
            self.ax_mean = PlasmaspherePlot._plot_polar_distribution(
                self.ax_mean, angle_values, l_values, mean_densities,
                colour_map=self.colour_map_density,
                title="Mean density, {} UTC".format(
                    self.date.strftime("%Y-%m-%d, %H:%M")),
                vmin=0,
                vmax=3.99

            )
            self.ax_mean = PlasmaspherePlot._add_colour_bar(
                self.ax_mean, self.colour_map_density,
                min_value=0, max_value=3.99, label='$log_{10}(n_e)$')

            ax_deviation = self.figure.add_subplot(grid[2:4, 0],
                                                   projection='polar')
            self.ax_deviation = ax_deviation
            deviation = PlasmaspherePlot._get_deviation(density_values.values)
            self.ax_deviation = PlasmaspherePlot._plot_polar_distribution(
                self.ax_deviation,
                angle_values,
                l_values,
                deviation,
                colour_map=self.colour_map_deviation,
                title="Standard deviation, {} UTC".format(
                    self.date.strftime("%Y-%m-%d, %H:%M")),
                vmin=0,
                vmax=1
            )
            self.ax_deviation = PlasmaspherePlot._add_colour_bar(
                self.ax_deviation, self.colour_map_deviation,
                min_value=0, max_value=1,
                label='$log_{10}(n_e)$')

        if df_kp is not None:
            self.ax_kp = self.figure.add_subplot(grid[0, 1:3])
            self.ax_kp.set_title('Kp')
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax_kp.xaxis.set_major_locator(locator)
            self.ax_kp.xaxis.set_major_formatter(formatter)
            kp_columns = [column for column in df_kp.columns
                          if "kp" in column]
            self.ax_kp = PlasmaspherePlot._plot_input(
                self.ax_kp, df_kp, time_column="t", input_columns=kp_columns
            )
            self.ax_kp.set_ylabel("Kp")
            self.ax_kp.set_xlabel("Time")
            self.ax_kp.set_ylim([0, 8])
            self.ax_kp.set_yticks([0, 2, 4, 6, 8])

        if df_solar_wind is not None:
            self.ax_solar_wind_Bz = self.figure.add_subplot(grid[1, 1:3])
            self.ax_solar_wind_Bz.set_title('Solar wind Bz')
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax_solar_wind_Bz.xaxis.set_major_locator(locator)
            self.ax_solar_wind_Bz.xaxis.set_major_formatter(formatter)
            bz_columns = [column for column in df_solar_wind.columns
                          if "bz_gsm" in column]
            self.ax_solar_wind_Bz = PlasmaspherePlot._plot_input(
                self.ax_solar_wind_Bz, df_solar_wind,
                time_column="t", input_columns=bz_columns
            )
            self.ax_solar_wind_Bz.set_ylabel("Bz")
            self.ax_solar_wind_Bz.set_xlabel("Time")

            self.ax_solar_wind_speed = self.figure.add_subplot(grid[2, 1:3])
            self.ax_solar_wind_speed.set_title('Solar wind speed')
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax_solar_wind_speed.xaxis.set_major_locator(locator)
            self.ax_solar_wind_speed.xaxis.set_major_formatter(formatter)
            speed_columns = [column for column in df_solar_wind.columns
                             if "speed" in column]
            self.ax_solar_wind_speed = PlasmaspherePlot._plot_input(
                self.ax_solar_wind_speed, df_solar_wind,
                time_column="t", input_columns=speed_columns
            )
            self.ax_solar_wind_speed.set_ylabel("Speed")
            self.ax_solar_wind_speed.set_xlabel("Time")

            self.ax_solar_wind_proton_density = self.figure.add_subplot(grid[3, 1:3])
            self.ax_solar_wind_proton_density.set_title("Solar wind proton density")
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax_solar_wind_proton_density.xaxis.set_major_locator(locator)
            self.ax_solar_wind_proton_density.xaxis.set_major_formatter(formatter)
            proton_density_columns = \
                [column for column in df_solar_wind.columns
                 if "proton_density" in column]
            self.ax_solar_wind_proton_density = PlasmaspherePlot._plot_input(
                self.ax_solar_wind_proton_density, df_solar_wind,
                time_column="t", input_columns=proton_density_columns
            )
            self.ax_solar_wind_proton_density.set_xlabel("Time")
            self.ax_solar_wind_proton_density.set_ylabel("Proton density")

        plt.tight_layout()

    @staticmethod
    def _save(path):
        plt.savefig(path)
        plt.close()

    @staticmethod
    def _get_density_values(df):
        if not isinstance(df, pd.DataFrame):
            msg = "data must be an instance of a pandas dataframe, " \
                  "instead it is of type {}".format(type(df))
            logging.error(msg)
            raise TypeError(msg)

        density_columns = [column for column in df.columns
                           if "predicted_densities" in column]

        return df[density_columns].copy()

    def plot_output(self, data, output_folder, video_file_name,
                    kp_inputs, solar_wind_inputs):
        """
        It produces a video form the output of the plasmasphere predictions.
        In case the predicted densities are nan for some date,
        only the Earth and the basic skeleton appear.

        :param data: instance of pandas DataFrame containing the output of the plasmasphere prediction modules
        :type data: pandas.DataFrame
        :param output_folder: output folder where to store the video, specify as an absolute path
        :type output_folder: str
        :param video_file_name: filename of the video, with extension .mp4
        :type video_file_name: str
        """

        if not isinstance(data, pd.DataFrame):
            msg = "data must be an instance of a pandas dataframe, " \
                  "instead it is of type {}".format(type(data))
            logging.error(msg)
            raise TypeError(msg)

        required_columns = ["L", "MLT", "t"]
        for column in required_columns:
            if column not in data.columns:
                msg = "column {} is missing".format(column)
                logging.error(msg)
                raise ValueError(msg)

        if not isinstance(data.iloc[0]["t"], datetime):
            msg = "values of date column must be datetime objects"
            logging.error(msg)
            raise TypeError(msg)

        if not os.path.isdir(output_folder):
            msg = "specified output_folder doesn't exist"
            logging.error(msg)
            raise FileNotFoundError(msg)

        output_folder = os.path.abspath(output_folder)

        if os.path.isfile(os.path.join(output_folder, video_file_name)):
            os.remove(os.path.join(output_folder, video_file_name))

        temp_folder_path = os.path.join(output_folder, "temp_with_inputs/")
        if os.path.exists(temp_folder_path):
            temp_folder_files = glob.glob(os.path.join(temp_folder_path, "*.png"))
            for file in temp_folder_files:
                os.remove(file)
        else:
            os.makedirs(temp_folder_path)

        logging.info("Started individual plasmasphere reconstructions generation")
        dates = pd.to_datetime(data["t"].unique())
        for date in dates:
            df_date = data[data["t"] == date]

            l_values = df_date["L"].values
            mlt_values = df_date["MLT"].values
            date = df_date.iloc[0]["t"]
            density_values = PlasmaspherePlot._get_density_values(df_date)
            if not PlasmaspherePlot._predictions_all_nan(density_values):
                df_kp_date = None
                solar_wind_date = None
                if kp_inputs is not None and solar_wind_inputs is not None:
                    df_kp_date = kp_inputs[
                        (kp_inputs["t"] <= date) &
                        (kp_inputs["t"] >= date - timedelta(days=3))
                    ]
                    solar_wind_date = solar_wind_inputs[
                        (solar_wind_inputs["t"] <= date) &
                        (solar_wind_inputs["t"] >= date - timedelta(days=3))
                    ]

                self._plot_single_date_prediction(l_values, mlt_values,
                                                  density_values, date,
                                                  df_kp=df_kp_date,
                                                  df_solar_wind=solar_wind_date)

                year, month, day, hour, minute = PlasmaspherePlot._get_date_components(date)
                self._save(os.path.join(temp_folder_path,
                                        "./plasmasphere_{}_{}_{}_{}_{}.png".format(
                                        year, month, day, hour, minute)
                                        )
                           )
                logging.info("for date {} plot has been generated".format(date))
            else:
                logging.warning("for date {} predictions were all NaN and "
                                "the plot has not been generated".format(date))
        logging.info("Finished individual plasmasphere reconstructions generation")

        logging.info("Starting video generation")

        if not len(os.listdir(temp_folder_path)) == 0:

            starting_working_directory = os.getcwd()
            os.chdir(temp_folder_path)
            subprocess.check_call([
                'ffmpeg', '-framerate', '3', '-i', os.path.join(temp_folder_path, "%*.png"), '-vcodec', 'libx264', '-crf',
                '28', '-pix_fmt', 'yuv420p',
                os.path.join(output_folder, video_file_name)
            ])
            logging.info("Finished video generation and saving")

            for file_name in glob.glob(os.path.join(temp_folder_path, "*.png")):
                os.remove(file_name)

            os.chdir(starting_working_directory)
        else:
            logging.info("All predictions were Nan and no movie has been generated")
