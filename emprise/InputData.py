# -*- coding: utf-8 -*-
"""
Module containing EMPRISE InputData class and sub-classes

Structural input data and timeseries profiles
"""

import pandas as pd
import numpy
import scipy.sparse
import math  # Used in myround
import networkx as nx


class InputData(object):
    """
    Class for input data storage and import
    """

    # Required fields for investment analysis input data
    # Default value = -1 means that it should be computed by the program

    def __init__(self, number_of_stages, reference_years):
        """
        Create InputData object with data and methods for import and
        processing of EMPRISE input data
        """

        self.general = None
        self.node = None
        self.branches = None
        self.generator_thermal = None
        self.generator_renewable = None
        self.consumer_conventional = None
        self.storage = None
        self.branch = None
        self.cost = None
        self.cost_generation = None
        self.cost_storage = None

        self.ts_consumer_conventional = None
        self.ts_generator_wind = None
        self.ts_generator_solar = None
        self.ts_periodWeightFactor = None

        self.number_of_stages = number_of_stages
        self.reference_years = reference_years

        self.time_delta = None
        self.time_range = None

        # Constants
        self.const_emission_factor = {
            "natural_gas": 56.0 * 3.6 / 1000.0,
            "uranium": 0.0,
        }  # tCO2eq/MWh_th

    def readStructuralData(self, file_path):
        """Read structural input data from files into data variables
        file_path: Dictionary with file paths to input data components (Example: file_path['nodes'] = '/data/nodes.csv')
        """

        self.general = {
            "finance": {
                "interestRate": 0.03,  # p.u. (annual interest rate)
            },
            "willingnessToPay": 500.0,  # EUR/MWh
            "yearsPerStage": 10,
        }

        self.node = pd.read_csv(
            file_path["nodes"],
            dtype={"id": str, "area": str, "lat": float, "lon": float},
        )
        self.branch = pd.read_csv(
            file_path["branches"],
            dtype={
                "node_from": str,
                "node_to": str,
                "capacity": float,
                "expand": bool,
                "type": str,
                "distance": float,
            },
        )

        self.generator_thermal = pd.read_csv(
            file_path["generators_thermal"],
            dtype={
                "node": str,
                "type": str,
                "fuel": str,
                "eta": float,
                "potential_max": str,
            },
        )
        self.generator_renewable = pd.read_csv(
            file_path["generators_renewable"],
            dtype={
                "node": str,
                "type": str,
                "iec": int,
                "lcoe": int,
                "potential_max": str,
            },
        )
        self.consumer_conventional = pd.read_csv(
            file_path["consumers_conventional"],
            dtype={"node": str, "annualDemand": float},
        )

        self.storage = pd.read_csv(
            file_path["storage"],
            dtype={
                "type": str,
                "node": str,
                "desc": str,
                "eta_out": float,
                "eta_in": float,
                "ratio_volume": float,
                "sdr": float,
                "dod": float,
                "potential_max": str,
            },
        )

        self.cost_generation = pd.read_csv(
            file_path["cost_generation"],
            dtype={
                "type": str,
                "subtype": str,
                "depreciation": str,
                "interest_rate": str,
                "capex": str,
                "opex_variable": str,
                "opex_fixed": str,
            },
        )

        self.cost_storage = pd.read_csv(
            file_path["cost_storage"],
            dtype={
                "type": str,
                "subtype": str,
                "depreciation": str,
                "interest_rate": str,
                "capex": str,
                "opex_variable": str,
                "opex_fixed": str,
            },
        )

        self.cost = {
            "systemOperation": {
                "fuel": {"natural_gas": 0.00003, "uranium": 0.000005},  # MEUR/MWh
                "emissionPrice": {stage: 0.0 for stage in range(1, self.number_of_stages + 1)},
            },
            "investment": {
                "capexThermal": {
                    "CCGT": 750000.0,
                    "OCGT": 420000.0,
                    "ST": 1000000.0,
                },  # EUR/MW # not used anymore (needs to be removed)
                "opexThermal": {
                    "CCGT": 30000.0,
                    "OCGT": 8000.0,
                    "ST": 15000.0,
                },  # EUR/MW/a # not used anymore (needs to be removed)
                "capexRenewable": {
                    "ONSHORE_IEC_1": 1100000.0,
                    "ONSHORE_IEC_3": 1400000.0,
                    "OFFSHORE_IEC_1": 3700000.0,
                    "SOLAR_ROOFTOP": 514000.0,
                    "SOLAR_UTILITY": 712000.0,
                },  # EUR/MW # not used anymore (needs to be removed)
                "opexRenewable": {
                    "ONSHORE_IEC_1": 17000.0,
                    "ONSHORE_IEC_3": 17000.0,
                    "OFFSHORE_IEC_1": 100000.0,
                    "SOLAR_ROOFTOP": 40000.0,
                    "SOLAR_UTILITY": 40000.0,
                },  # EUR/MW/a # not used anymore (needs to be removed)
            },
        }

    def _readTimeseriesFromFile(self, filename, n_header_rows, tuples, reference_years, timerange):
        for t in tuples:
            if len(t) == 2:  # conventional load
                tmp_filename = filename.replace("%NODE%", t[0]).replace("%REFYEAR%", t[1])
            elif len(t) == 4:  # wind, solar
                tmp_filename = filename.replace("%NODE%", t[0]).replace("%TYPE%", t[1]).replace("%LCOE%", t[2]).replace("%REFYEAR%", t[3])
            else:
                print("### Cannot process timeseries tuple information: ", t, " ###")

            if "ts" not in locals():
                ts = pd.read_csv(tmp_filename, header=n_header_rows)
            else:
                ts = pd.concat([ts, pd.read_csv(tmp_filename, header=n_header_rows)], axis=1)

        pwf = len(ts.index) / len(timerange)
        ts = ts.loc[timerange, tuples]
        ts.index = range(len(timerange))
        return (ts, pwf)

    def readTimeseriesData(self, file_names, n_header_rows, reference_years, timerange, timedelta=1.0):
        """Read timeseries data into numpy arrays"""
        import itertools

        tuples_conventional = [(n, r) for n, r in itertools.product(self.node["id"], [str(ry) for ry in set(reference_years)])]
        (self.ts_consumer_conventional, self.ts_periodWeightFactor,) = self._readTimeseriesFromFile(
            file_names["consumers_conventional"],
            n_header_rows["consumers_conventional"],
            tuples_conventional,
            set(reference_years),
            timerange,
        )

        tuples_solar = [
            (n[0], n[1], n[2], r)
            for n, r in itertools.product(
                [(row["node"], row["type"], "LCOE_" + str(row["lcoe"])) for k, row in self.generator_renewable.iterrows() if row["type"].startswith("SOLAR")],
                [str(ry) for ry in set(reference_years)],
            )
        ]
        (self.ts_generator_solar, self.ts_periodWeightFactor,) = self._readTimeseriesFromFile(
            file_names["solar"],
            n_header_rows["solar"],
            tuples_solar,
            set(reference_years),
            timerange,
        )

        tuples_wind = [
            (n[0], n[1], n[2], r)
            for n, r in itertools.product(
                [(row["node"], row["type"], "LCOE_" + str(row["lcoe"])) for k, row in self.generator_renewable.iterrows() if (row["type"].startswith("ONSHORE") or row["type"].startswith("OFFSHORE"))],
                [str(ry) for ry in set(reference_years)],
            )
        ]
        (self.ts_generator_wind, self.ts_periodWeightFactor,) = self._readTimeseriesFromFile(
            file_names["wind"],
            n_header_rows["wind"],
            tuples_wind,
            set(reference_years),
            timerange,
        )

        self.time_range = timerange
        self.time_delta = timedelta

        return

    def getAllAreas(self):
        """Return list of areas included in the model"""
        areas = self.node["area"]
        all_areas = []
        for area in areas:
            if area not in all_areas:
                all_areas.append(area)
        return all_areas
