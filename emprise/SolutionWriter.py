# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 2021

@author: Philipp Härtel
"""

from pyomo.common.plugin import *
from pyomo.pysp import solutionwriter
from pyomo.pysp.scenariotree import *
from pyomo.pysp.plugins.phhistoryextension import (
    extract_scenario_tree_structure,
    extract_scenario_solutions,
    extract_node_solutions,
)
import os

import json


class SolutionWriter(SingletonPlugin):

    implements(solutionwriter.ISolutionWriterExtension)

    def write(self, scenario_tree, output_file_prefix):

        if not isinstance(scenario_tree, ScenarioTree):
            raise RuntimeError("SolutionWriter write method expects ScenarioTree object - type of supplied object=" + str(type(scenario_tree)))

        include_ph_objective_parameters = None
        include_variable_statistics = None
        if output_file_prefix == "ph":
            include_ph_objective_parameters = True
            include_variable_statistics = True
        elif output_file_prefix == "postphef":
            include_ph_objective_parameters = False
            include_variable_statistics = True
        elif output_file_prefix == "ef":
            include_ph_objective_parameters = False
            include_variable_statistics = False
        else:
            raise ValueError("JSONSolutionWriter requires an output prefix of 'ef', 'ph', or 'postphef' " "to indicate whether ph specific parameter values should be extracted " "from the solution")

        if os.name == "nt":
            result_dir = os.path.join(os.getcwd(), "result")
        else:
            result_dir = os.path.join(os.getcwd(), "python", "simulations", "result")
            if "EMPRISE_JOB_NAME" in os.environ:
                result_dir = os.path.join(result_dir, os.environ["EMPRISE_JOB_NAME"])

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        output_filename_tree = output_file_prefix + "_scenario_tree_structure.json"
        output_filename_scenario = output_file_prefix + "_scenario_solution.json"
        output_filename_scenario_cost = output_file_prefix + "_scenario_solution_cost.json"
        output_filename_scenario_capacity = output_file_prefix + "_scenario_solution_capacity.json"
        output_filename_scenario_exchange = output_file_prefix + "_scenario_solution_exchange.json"
        output_filename_scenario_generation_thermal = output_file_prefix + "_scenario_solution_generation_thermal.json"
        output_filename_scenario_generation_renewable = output_file_prefix + "_scenario_solution_generation_renewable.json"
        output_filename_scenario_curtailment_renewable = output_file_prefix + "_scenario_solution_curtailment_renewable.json"
        output_filename_node = output_file_prefix + "_node_solution.json"

        # --- Retrieve scenario tree structure information
        results_scenario_tree_structure = extract_scenario_tree_structure(scenario_tree)
        print("\n### Scenario tree structure information retrieved! ###")

        # --- Retrieve scenario solutions
        results_scenario_solutions = extract_scenario_solutions(
            scenario_tree,
            include_ph_objective_parameters=include_ph_objective_parameters,
        )
        print("### Scenario solutions retrieved! ###")

        scenario_keys = list(results_scenario_solutions.keys())
        scenario_element_keys = list(results_scenario_solutions[scenario_keys[0]].keys())
        scenario_var_keys = set([self.replace_between(var_name, "[", "]") for var_name in list(results_scenario_solutions[scenario_keys[0]][scenario_element_keys[0]].keys())])

        results_scenario_solutions_cost = {}
        results_scenario_solutions_capacity = {}
        results_scenario_solutions_exchange = {}
        results_scenario_solutions_generation_thermal = {}
        results_scenario_solutions_generation_renewable = {}
        results_scenario_solutions_curtailment_renewable = {}

        for sc in scenario_keys:
            results_scenario_solutions_cost[sc] = results_scenario_solutions[sc].copy()
            results_scenario_solutions_capacity[sc] = results_scenario_solutions[sc]["variables"].copy()
            results_scenario_solutions_exchange[sc] = results_scenario_solutions[sc]["variables"].copy()
            results_scenario_solutions_generation_thermal[sc] = results_scenario_solutions[sc]["variables"].copy()
            results_scenario_solutions_generation_renewable[sc] = results_scenario_solutions[sc]["variables"].copy()
            results_scenario_solutions_curtailment_renewable[sc] = results_scenario_solutions[sc]["variables"].copy()

            tmp_var_names = list(results_scenario_solutions_cost[sc]["variables"].keys())
            for var_name in tmp_var_names:
                # Extract cost results from scenario solution
                if not var_name.startswith("cost"):
                    results_scenario_solutions_cost[sc]["variables"].pop(var_name, None)
                # else:
                # results_scenario_solutions_cost[sc][var_name].pop("fixed", None)
                # results_scenario_solutions_cost[sc][var_name].pop("stale", None)
                # results_scenario_solutions_cost[sc][var_name] = results_scenario_solutions_cost[sc][var_name].pop("value")

                # Extract capacity results from scenario solution
                if not any(
                    [
                        x in var_name
                        for x in [
                            "generatorThermalCapacity",
                            "generatorThermalNewCapacity",
                            "generatorThermalDecommissionedCapacity",
                            "generatorRenewableCapacity",
                            "generatorRenewableNewCapacity",
                            "generatorRenewableDecommissionedCapacity",
                        ]
                    ]
                ):
                    results_scenario_solutions_capacity[sc].pop(var_name, None)
                else:
                    results_scenario_solutions_capacity[sc][var_name].pop("fixed", None)
                    results_scenario_solutions_capacity[sc][var_name].pop("stale", None)
                    results_scenario_solutions_capacity[sc][var_name].pop("rho", None)
                    results_scenario_solutions_capacity[sc][var_name].pop("weight", None)
                    results_scenario_solutions_capacity[sc][var_name] = results_scenario_solutions_capacity[sc][var_name].pop("value")

                # Extract exchange results from scenario solution
                if not any([x in var_name for x in ["flow"]]):
                    results_scenario_solutions_exchange[sc].pop(var_name, None)
                else:
                    results_scenario_solutions_exchange[sc][var_name].pop("fixed", None)
                    results_scenario_solutions_exchange[sc][var_name].pop("stale", None)
                    results_scenario_solutions_exchange[sc][var_name].pop("rho", None)
                    results_scenario_solutions_exchange[sc][var_name].pop("weight", None)
                    results_scenario_solutions_exchange[sc][var_name] = results_scenario_solutions_exchange[sc][var_name].pop("value")

                # Extract thermal generation results from scenario solution
                if not any(
                    [
                        x in var_name
                        for x in [
                            "generatorThermalCapacity",
                            "generatorThermalNewCapacity",
                            "generationThermal",
                        ]
                    ]
                ):
                    results_scenario_solutions_generation_thermal[sc].pop(var_name, None)
                else:
                    results_scenario_solutions_generation_thermal[sc][var_name].pop("fixed", None)
                    results_scenario_solutions_generation_thermal[sc][var_name].pop("stale", None)
                    results_scenario_solutions_generation_thermal[sc][var_name].pop("rho", None)
                    results_scenario_solutions_generation_thermal[sc][var_name].pop("weight", None)
                    # results_scenario_solutions_generation_thermal[sc][var_name] = results_scenario_solutions_generation_thermal[sc][var_name].pop("value")

                # Extract renewable generation results from scenario solution
                if not any(
                    [
                        x in var_name
                        for x in [
                            "generatorRenewableCapacity",
                            "generatorRenewableNewCapacity",
                            "generationRenewable",
                        ]
                    ]
                ):
                    results_scenario_solutions_generation_renewable[sc].pop(var_name, None)
                else:
                    results_scenario_solutions_generation_renewable[sc][var_name].pop("fixed", None)
                    results_scenario_solutions_generation_renewable[sc][var_name].pop("stale", None)
                    results_scenario_solutions_generation_renewable[sc][var_name].pop("rho", None)
                    results_scenario_solutions_generation_renewable[sc][var_name].pop("weight", None)
                    # results_scenario_solutions_generation_renewable[sc][var_name] = results_scenario_solutions_generation_renewable[sc][var_name].pop("value")

                # Extract renewable curtailment results from scenario solution
                if not any([x in var_name for x in ["curtailmentRenewable"]]):
                    results_scenario_solutions_curtailment_renewable[sc].pop(var_name, None)
                else:
                    results_scenario_solutions_curtailment_renewable[sc][var_name].pop("fixed", None)
                    results_scenario_solutions_curtailment_renewable[sc][var_name].pop("stale", None)
                    results_scenario_solutions_curtailment_renewable[sc][var_name].pop("rho", None)
                    results_scenario_solutions_curtailment_renewable[sc][var_name].pop("weight", None)

        del results_scenario_solutions

        # --- Retrieve node solutions
        results_node_solutions = extract_node_solutions(
            scenario_tree,
            include_ph_objective_parameters=include_ph_objective_parameters,
            include_variable_statistics=include_variable_statistics,
        )
        print("### Node solutions retrieved! ###")

        node_keys = list(results_node_solutions.keys())
        # node_element_keys = list(results_node_solutions[node_keys[0]].keys())

        results_node_solutions_cost = {}
        for n in node_keys:
            # Extract cost results from node solution
            results_node_solutions_cost[n] = results_node_solutions[n].copy()
            results_node_solutions_cost[n].pop("variables", None)
        del results_node_solutions

        results_solutions_cost = {
            "scenario": results_scenario_solutions_cost,
            "node": results_node_solutions_cost,
        }

        # --- Write output data to result files
        with open(os.path.join(result_dir, output_filename_tree), "w") as f:
            json.dump(results_scenario_tree_structure, f, indent=2)
        print("### Scenario tree structure written to file=" + output_filename_tree + " ###")

        # with open(os.path.join(os.getcwd(), "result", output_filename_scenario), 'w') as f:
        #     json.dump(results_scenario_solutions, f, indent=2)
        # print("Scenario solutions written to file=" + output_filename_scenario)

        with open(os.path.join(result_dir, output_filename_scenario_exchange), "w") as f:
            json.dump(results_scenario_solutions_exchange, f, indent=2)
        print("### Scenario exchange solutions written to file=" + output_filename_scenario_exchange + " ###")

        with open(os.path.join(result_dir, output_filename_scenario_capacity), "w") as f:
            json.dump(results_scenario_solutions_capacity, f, indent=2)
        print("### Scenario capacity solutions written to file=" + output_filename_scenario_capacity + " ###")

        with open(os.path.join(result_dir, output_filename_scenario_generation_thermal), "w") as f:
            json.dump(results_scenario_solutions_generation_thermal, f, indent=2)
        print("### Scenario generation thermal solutions written to file=" + output_filename_scenario_generation_thermal + " ###")

        with open(os.path.join(result_dir, output_filename_scenario_generation_renewable), "w") as f:
            json.dump(results_scenario_solutions_generation_renewable, f, indent=2)
        print("### Scenario generation renewable solutions written to file=" + output_filename_scenario_generation_renewable + " ###")

        with open(os.path.join(result_dir, output_filename_scenario_curtailment_renewable), "w") as f:
            json.dump(results_scenario_solutions_curtailment_renewable, f, indent=2)
        print("### Scenario curtailment renewable solutions written to file=" + output_filename_scenario_curtailment_renewable + " ###")

        with open(os.path.join(result_dir, output_filename_scenario_cost), "w") as f:
            json.dump(results_solutions_cost, f, indent=2)
        print("### Scenario cost solutions written to file=" + output_filename_scenario_cost + " ###")

        # with open(os.path.join(os.getcwd(), "result", output_filename_node), 'w') as f:
        #     json.dump(results_node_solutions, f, indent=2)
        # print("Node solutions written to file=" + output_filename_node)

    def replace_between(self, text, begin, end, alternative=""):
        middle = text.split(begin, 1)[1].split(end, 1)[0]
        return text.replace(begin + middle + end, alternative)
