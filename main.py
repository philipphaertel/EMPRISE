# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 2021

@author: Philipp Härtel
"""
import os
import emprise
import numpy as np
import json

from pyomo.common.tempfiles import TempfileManager

import pyomo.environ as pyo

# pyo.pyomo.common.timing.report_timing()

# Scenario (tree) configuration
scenario_name = "test"
scenario_variant = "storage"
# scenario_variant = "det_inv_high"

exclude_component_list = ["electricity_storage"]  # []  # ["electricity_storage"], []

sample_size = 6 * 30 * 24  # hourly time steps of system operation sample
sample_offset = 26 * 7 * 24  # offset of system operation sample
reference_years = [2012]  # [2010, 2012]  # [2006, 2008]  # [2007, 2008, 2011]

number_of_stages = 3  # Planning periods, e.g. 3 (representing planning periods 2025, 2035, 2045)
number_of_investment_scenarios = 2
number_of_system_operation_scenarios = len(reference_years)

scenario_name_extended = scenario_name + "_" + scenario_variant + "_" + "_".join([str(ry) for ry in reference_years])

# Create scenario probability dictionary
scenario_probability_sysop = {"sysop_" + str(i): [1.0 / number_of_system_operation_scenarios] * number_of_system_operation_scenarios for i in range(1, number_of_stages + 1)}
scenario_probability_inv = {"inv_" + str(i): [0.7, 0.3] for i in range(1, number_of_stages + 1)}
# scenario_probability_inv = {"inv_" + str(i): [1.0] for i in range(1, number_of_stages + 1)}
scenario_probability = {**scenario_probability_inv, **scenario_probability_sysop}

# --- Path and directory information
data_folder_name = "data"
plot_folder_name = "plot"
result_folder_name = "result/" + scenario_name
json_file_name = "ef_solution.json"
cluster_working_directory = "/home/phaertel/python/simulations"

print("### Initializing EMPRISE framework for scenario setup '" + scenario_name_extended + "' ###")


def checkAndCreateDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


# --- Check whether running on local windows or other machine (HPCC)
if os.name == "nt":
    working_dir = ""
    os.environ["EMPRISE_JOB_NAME"] = scenario_name_extended
else:
    working_dir = cluster_working_directory
    if "TMP_PYOMO_DIR" in os.environ:
        cluster_tmp_directory = os.environ["TMP_PYOMO_DIR"]
        TempfileManager.tempdir = cluster_tmp_directory
        # print('### Using temporary pyomo directory ', cluster_tmp_directory, ' ###')

path_data_dir = os.path.join(working_dir, data_folder_name)
path_plot_dir = checkAndCreateDirectory(os.path.join(working_dir, plot_folder_name))
path_result_dir = checkAndCreateDirectory(os.path.join(working_dir, result_folder_name))
path_json_dir = os.path.join(path_result_dir, json_file_name)

# --- Strucural input data directory information
file_names_structural = {
    "nodes": os.path.join(path_data_dir, scenario_name + "_nodes.csv"),
    "branches": os.path.join(path_data_dir, scenario_name + "_branches.csv"),
    "generators_thermal": os.path.join(path_data_dir, scenario_name + "_generators_thermal.csv"),
    "generators_renewable": os.path.join(path_data_dir, scenario_name + "_generators_renewable.csv"),
    "consumers_conventional": os.path.join(path_data_dir, scenario_name + "_consumers_conventional.csv"),
    "storage": os.path.join(path_data_dir, scenario_name + "_storage.csv"),
    "cost_generation": os.path.join(path_data_dir, scenario_name + "_cost_generation.csv"),
    "cost_storage": os.path.join(path_data_dir, scenario_name + "_cost_storage.csv"),
}

# --- Timeseries input data directory information
file_names_timeseries = {
    "consumers_conventional": os.path.join(
        path_data_dir,
        "ts",
        scenario_name + "_consumers_conventional_%NODE%_%REFYEAR%_timeseries.csv",
    ),
    "wind": os.path.join(
        path_data_dir,
        "ts",
        scenario_name + "_wind_%NODE%_%TYPE%_%LCOE%_%REFYEAR%_timeseries.csv",
    ),
    "solar": os.path.join(
        path_data_dir,
        "ts",
        scenario_name + "_solar_%NODE%_%TYPE%_%LCOE%_%REFYEAR%_timeseries.csv",
    ),
}

n_header_rows_timeseries = {
    "consumers_conventional": [0, 1],
    "wind": [0, 1, 2, 3],
    "solar": [0, 1, 2, 3],
}

# Create abstract model
emprise_model = emprise.EmpriseModel(exclude_component_list=exclude_component_list)

if __name__ == "__main__":
    # --- Retrieve EMPRISE input data
    print("### Retrieving EMPRISE structural input data ...", end="", flush=True)

    input_data = emprise.InputData(number_of_stages, reference_years)
    input_data.readStructuralData(file_path=file_names_structural)

    print(" done! ###")

    print("### Retrieving EMPRISE timeseries input data ...", end="", flush=True)

    input_data.readTimeseriesData(
        file_names=file_names_timeseries,
        n_header_rows=n_header_rows_timeseries,
        reference_years=reference_years,
        timerange=range(sample_offset, sample_size + sample_offset),
        timedelta=1.0,
    )

    print(" done! ###")

    # --- Create EMPRISE model instance
    print("### Creating EMPRISE model instance ...", end="", flush=True)

    dict_data = emprise_model.createModelData(input_data)

    print(" done! ###")


# --- Define scenario modification function


def scenario_data(dict_data_sc, scenario_name, inv_sc, sysop_sc):
    """Modify dictionary data according to the scenario setup"""
    # dict_data_sc = emprise_model.updateOperationalScenarioInformation(dict_data, input_data, sysop_sc, reference_years)

    if number_of_stages == 4:
        emission_price = np.matrix([[20.0, 30.0, 90.0, 120.0], [40.0, 80.0, 130.0, 300.0]])
    elif number_of_stages == 3:
        emission_price = np.matrix([[70.0, 100.0, 200.0], [70.0, 70.0, 70.0]])
        # emission_price = np.matrix([[70.0, 100.0, 200.0]])
        # emission_price = np.matrix([[80.0, 100.0, 200.0], [80.0, 200.0, 350.0]])
    elif number_of_stages == 2:
        emission_price = np.matrix([[20.0, 30.0], [20.0, 80.0]])

    dict_data_sc["emprise"]["costSystemOperationEmissionPrice"] = {stage: emission_price[inv_sc[stage - 1] - 1, stage - 1] for stage in range(1, dict_data_sc["emprise"]["numberOfStages"][None] + 1)}  # EUR/tCO2eq
    print(dict_data_sc["emprise"]["costSystemOperationEmissionPrice"])

    return dict_data_sc


# --- Define stochastic callback functions
def pysp_scenario_tree_model_callback():
    """Call-back function to create scenario tree model (alternative to using ScenarioStructure.dat file input)"""

    print("### Creation of EMPRISE scenario graph model started! ###")

    scenario_tree_graph = emprise_model.createScenarioTreeGraph(  # not 2 (explicit tsm not necessary anymore)
        n_stages=number_of_stages,
        n_inv_sc=number_of_investment_scenarios,
        n_sysop_sc=number_of_system_operation_scenarios,
        scenario_probability=scenario_probability,
        path_result_dir=path_result_dir,
    )

    # scenario_tree_graph.pprint(os.path.join(path_result_dir, 'st_scenario_tree_model_{}.txt'.format(scenario_name)))

    print("### Creation of EMPRISE scenario graph model finished! ###")
    return scenario_tree_graph


def pysp_instance_creation_callback(scenario_tree_model, scenario_name, node_names):
    """Call-back function to create model instance"""
    scenario_info = scenario_name.strip("sc_")
    scenario_info = [int(x) for x in scenario_info.split("_")]
    stage = scenario_info[0]
    node = scenario_info[1]

    # print("tree model", scenario_tree_model, "scenario name", scenario_name, "node name", node_names)
    (sysop_sc, inv_sc) = emprise.EmpriseModel.parseScenarioInformation(
        stage,
        node,
        number_of_stages + 1,
        number_of_investment_scenarios,
        number_of_system_operation_scenarios,
    )  # stage  # scenario tree node number of that stage

    print(
        "### Creating instance for scenario {} ...".format(scenario_name),
        end="",
        flush=True,
    )

    scenario_specific_reference_years = [reference_years[sc - 1] for sc in sysop_sc[1:]]

    input_data = emprise.InputData(number_of_stages, scenario_specific_reference_years)
    input_data.readStructuralData(file_path=file_names_structural)

    input_data.readTimeseriesData(
        file_names=file_names_timeseries,
        n_header_rows=n_header_rows_timeseries,
        reference_years=scenario_specific_reference_years,
        timerange=range(sample_offset, sample_size + sample_offset),
        timedelta=1.0,
    )

    dict_data = emprise_model.createModelData(input_data)
    dict_data = scenario_data(dict_data, scenario_name, inv_sc, sysop_sc)

    instance = emprise_model.createConcreteModel(dict_data=dict_data)
    instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # if scenario_name.endswith("_1"):
    #     instance.pprint(os.path.join(path_result_dir, "st_data_{}.txt".format(scenario_name)))
    # instance.pprint()

    print(" done! ###")

    return instance


# --- Plot EMPRISE results from solution file (if run as a main)
if __name__ == "__main__":

    print(
        "\n### Loading Progressive Hedging solution from '{}' ...".format(path_json_dir),
        end="",
        flush=True,
    )
    with open(path_json_dir) as handle:
        result_ph = json.loads(handle.read())
    print(" done! ###")

    print("### Creating EMPRISE model instance ...", end="", flush=True)
    model = emprise_model.createConcreteModel(dict_data=dict_data)
    print(" done! ###")

    print("### Retrieving EMPRISE result data from solution ...", end="", flush=True)
    result_data = emprise.ResultData(input_data)
    result_data = emprise_model.extractResultData(result_data, result_ph, model)
    print(" done! ###")

    print("### Plotting of EMPRISE results started! ###")
    result_data.plotResults(path_plot_dir, scenario_name)
    print("### Plotting of EMPRISE results finished! ###")
