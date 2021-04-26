# -*- coding: utf-8 -*-
"""
Module containing EMPRISE ResultData class and sub-classes

Structural result data and timeseries profiles
"""

import pandas as pd
import numpy as np
import scipy.sparse
import math  # Used in myround
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class ResultData(object):
    """
    Class for EMPRISE result data storage and import
    """

    def __init__(self, input_data):
        """
        Create ResultData object with data and methods for import and
        processing of EMPRISE result data
        """

        self.general = input_data.general
        self.node = input_data.node
        self.generator_thermal = input_data.generator_thermal
        self.generator_renewable = input_data.generator_renewable
        self.consumer_conventional = input_data.consumer_conventional
        self.branch = input_data.branch
        self.cost = input_data.cost

        self.ts_consumer_conventional = input_data.ts_consumer_conventional
        self.ts_generator_wind = input_data.ts_generator_wind
        self.ts_generator_solar = input_data.ts_generator_solar
        self.ts_periodWeightFactor = input_data.ts_periodWeightFactor

        self.result_generation_thermal_capacity = None
        self.result_generation_thermal_capacity_expansion = None
        self.result_generation_renewable_capacity = None
        self.result_generation_renewable_capacity_expansion = None

        self.result_cost_investment = None
        self.result_cost_system_operation = None

        self.number_of_stages = input_data.number_of_stages

        self.time_delta = input_data.time_delta
        self.time_range = input_data.time_range

        # Constants
        self.const_emission_factor = input_data.const_emission_factor

    def getAllAreas(self):
        """Return list of areas included in the model"""
        areas = self.node["area"]
        allareas = []
        for area in areas:
            if area not in allareas:
                allareas.append(area)
        return allareas

    def plotResults(self, path_output_plot, scenario_name):
        """Plot all results"""
        import os

        self.plotClusteredStacked(
            [self.result_generation_thermal_capacity.loc[:, st].unstack() / 1000 for st in range(1, self.number_of_stages + 1)],
            ["Stage" + str(st) for st in range(1, self.number_of_stages + 1)],
            title="Thermal generation capacity",
            y_label="Installed capacity in GW",
        )
        plt.savefig(
            os.path.join(path_output_plot, scenario_name + "_cap_gen_thermal.pdf"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(path_output_plot, scenario_name + "_cap_gen_thermal.emf"),
            bbox_inches="tight",
        )
        plt.close()

        self.plotClusteredStacked(
            [self.result_generation_renewable_capacity.loc[:, st].unstack() / 1000 for st in range(1, self.number_of_stages + 1)],
            ["Stage" + str(st) for st in range(1, self.number_of_stages + 1)],
            title="Renewable generation capacity",
            y_label="Installed capacity in GW",
        )
        plt.savefig(
            os.path.join(path_output_plot, scenario_name + "_cap_gen_renewable.pdf"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(path_output_plot, scenario_name + "_cap_gen_renewable.emf"),
            bbox_inches="tight",
        )
        # plt.savefig(os.path.join(path_output_plot, scenario_name + '_cap_gen_renewable.png'), bbox_inches='tight')
        plt.close()

        self.plotClusteredStacked(
            [self.result_generation_thermal_capacity_expansion.loc[:, st].unstack() / 1000 for st in range(1, self.number_of_stages + 1)],
            ["Stage" + str(st) for st in range(1, self.number_of_stages + 1)],
            title="Thermal generation capacity expansion",
            y_label="Installed capacity in GW",
        )
        plt.savefig(
            os.path.join(path_output_plot, scenario_name + "_cap_gen_thermal_exp.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        self.plotClusteredStacked(
            [self.result_generation_renewable_capacity_expansion.loc[:, st].unstack() / 1000 for st in range(1, self.number_of_stages + 1)],
            ["Stage" + str(st) for st in range(1, self.number_of_stages + 1)],
            title="Renewable generation capacity expansion",
            y_label="Installed capacity in GW",
        )
        plt.savefig(
            os.path.join(path_output_plot, scenario_name + "_cap_gen_renewable_exp.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        # self.result_cost_system_operation.apply(lambda x: x / 1000000000).boxplot()
        self.result_cost_system_operation.apply(lambda x: x / 1000).hist()
        plt.savefig(
            os.path.join(path_output_plot, scenario_name + "_cost_system_operation_total.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        self.result_cost_investment.apply(lambda x: x / 1000).plot(kind="bar", linewidth=0)
        plt.savefig(
            os.path.join(path_output_plot, scenario_name + "_cost_investment_total.pdf"),
            bbox_inches="tight",
        )
        plt.close()

    def plotClusteredStacked(self, dfall, labels=None, title="multiple stacked bar plot", y_label="", H="/", **kwargs):
        """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
        labels is a list of the names of the dataframe, used for the legend
        title is a string for the title of the plot
        H is the hatch used for identification of the different dataframe
        https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
        """
        plt.rcParams.update(
            {
                "font.size": 12,
                "text.usetex": False,
                "axes.linewidth": 0.5,
                "hatch.linewidth": 0.25,
                "grid.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "legend.frameon": False,
                "figure.figsize": [31.2, 13.98],
            }
        )  # , 'font.name': 'computer modern roman'})

        # print(plt.rcParams.keys())
        n_df = len(dfall)
        n_col = len(dfall[0].columns)
        n_ind = len(dfall[0].index)

        axe = plt.subplot(111)

        for i, df in enumerate(dfall):  # for each data frame
            axe = df.plot(kind="bar", linewidth=0, stacked=True, ax=axe, legend=False, **kwargs)  # make bar plots
            axe.set_xlim([-0.5, n_ind - 0.25])

        axe.grid(b=True, which="major", axis="y")
        axe.set_axisbelow(True)

        h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
        for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
            for j, pa in enumerate(h[i : i + n_col]):
                for rect in pa.patches:  # for each index
                    rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                    rect.set_hatch(H * int(i / n_col))  # edited part
                    rect.set_width(1 / float(n_df + 1))

        axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0)
        axe.set_xticklabels(df.index, rotation=0)
        axe.set_title(title)
        axe.set_ylabel(y_label)

        # Add invisible data to add another legend
        n = []
        for i in range(n_df):
            n.append(axe.bar(0, 0, color="gray", hatch=H * i))

        h1 = h[:n_col]
        lab1 = l[:n_col]
        l1 = axe.legend(h1[::-1], lab1[::-1], loc=[1.01, 0.5])
        if labels is not None:
            l2 = plt.legend(n, labels, loc=[1.01, 0.0])
        axe.add_artist(l1)
        return axe
