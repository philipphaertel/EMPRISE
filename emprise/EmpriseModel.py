# -*- coding: utf-8 -*-
"""
Created on Wed February 10 2021

@author: Philipp Härtel
"""

import pyomo.environ as pyo
import networkx
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt


class EmpriseModel:
    """
    Environment for Modelling and Planning Robust Investments in Sector-integrated Energy systems (EMPRISE) Model
    """

    # _NUMERICAL_THRESHOLD_ZERO = 1e-6
    # _HOURS_PER_YEAR = 8760

    def __repr__(self):
        return f"EmpriseModel()"

    def __str__(self):
        return f"Environment for Modelling and Planning Robust Investments in Sector-integrated Energy systems (EMPRISE) Model"

    def __init__(self, exclude_component_list=[]):
        """
        Create Abstract Pyomo model for the EMPRISE problem
        """
        self.exclude_components = self._parseExludeComponents(exclude_component_list)
        self.abstract_model = self._createAbstractModel()

    def _parseExludeComponents(self, exclude_component_list):
        exclude_components = {"electricity_storage": False, "thermal_generation": False, "renewable_generation": False}
        if any("electricity_storage" in s for s in exclude_component_list):
            exclude_components["electricity_storage"] = True

        return exclude_components

    def _createAbstractModel(self):
        model = pyo.AbstractModel()
        model.name = "EMPRISE abstract model"

        model.numberOfStages = pyo.Param(within=pyo.NonNegativeIntegers)

        # Sets ############################################################################################################################

        model.STAGE_MODEL = pyo.RangeSet(model.numberOfStages + 1)
        model.STAGE = pyo.RangeSet(model.numberOfStages)  # , within=model.STAGE_MODEL)

        def stageOperational_init(model):
            return [(x, y) for x in model.STAGE for y in model.STAGE if x <= y]

        model.STAGE_OPERATIONAL = pyo.Set(initialize=stageOperational_init, within=model.STAGE * model.STAGE)

        def stageDecommissioning_init(model):
            return [(x, y) for x in model.STAGE for y in model.STAGE if x < y]

        model.STAGE_DECOMMISSIONING = pyo.Set(initialize=stageDecommissioning_init, within=model.STAGE * model.STAGE)

        model.FUEL = pyo.Set()
        model.LOAD = pyo.Set()
        model.AREA = pyo.Set()
        model.TIME = pyo.Set()

        model.NODE = pyo.Set()  # Example: DEU1

        model.GEN_THERMAL = pyo.Set()
        model.GEN_THERMAL_TYPE = pyo.Set()  # Example: OCGT
        model.GEN_RENEWABLE = pyo.Set()
        model.GEN_RENEWABLE_TYPE = pyo.Set()  # Example: ONSHORE_IEC_1
        model.GEN_TYPE = model.GEN_THERMAL_TYPE | model.GEN_RENEWABLE_TYPE
        model.STORAGE = pyo.Set()
        model.STORAGE_TYPE = pyo.Set()  # Example: LI_ION

        model.BRANCH = pyo.Set(dimen=2)

        def nodesOut_init(model, node):
            retval = []
            for (i, j) in model.BRANCH:
                if i == node:
                    retval.append(j)
            return retval

        model.NODE_OUT = pyo.Set(model.NODE, initialize=nodesOut_init)

        def nodesIn_init(model, node):
            retval = []
            for (i, j) in model.BRANCH:
                if j == node:
                    retval.append(i)
            return retval

        model.NODE_IN = pyo.Set(model.NODE, initialize=nodesIn_init)

        # Parameters ######################################################################################################################
        # - General
        # model.samplefactor = pyo.Param(model.TIME, within=pyo.NonNegativeReals)
        model.financePresentValueInterestRate = pyo.Param(within=pyo.Reals)
        model.financeDeprecationPeriodGenerationThermal = pyo.Param(model.GEN_THERMAL_TYPE, within=pyo.NonNegativeIntegers)
        model.financeDeprecationPeriodGenerationRenewable = pyo.Param(model.GEN_RENEWABLE_TYPE, within=pyo.NonNegativeIntegers)
        model.willingnessToPay = pyo.Param(within=pyo.NonNegativeReals)
        model.yearsPerStage = pyo.Param(within=pyo.NonNegativeIntegers)
        model.periodWeightFactor = pyo.Param(within=pyo.NonNegativeReals)
        model.emissionFactor = pyo.Param(model.FUEL, within=pyo.NonNegativeReals)

        model.operationalStageContributionGeneration = pyo.Param(
            model.STAGE_OPERATIONAL, model.STAGE, model.GEN_THERMAL_TYPE | model.GEN_RENEWABLE_TYPE, within=pyo.PercentFraction
        )  # model.STAGE represents the technology development stages
        model.operationalStageContributionStorage = pyo.Param(model.STAGE_OPERATIONAL, model.STAGE, model.STORAGE_TYPE, within=pyo.PercentFraction)  # model.STAGE represents the technology development stages

        # - Generation thermal
        model.generationThermalNode = pyo.Param(model.GEN_THERMAL, within=model.NODE)
        model.generationThermalType = pyo.Param(model.GEN_THERMAL, within=model.GEN_THERMAL_TYPE)
        model.generationThermalFuel = pyo.Param(model.GEN_THERMAL, within=model.FUEL)
        model.generationThermalEta = pyo.Param(model.GEN_THERMAL, within=pyo.NonNegativeReals)
        model.generationThermalPotentialMax = pyo.Param(model.STAGE, model.GEN_THERMAL, within=pyo.Reals)

        # - Generation renewable
        model.generationRenewableNode = pyo.Param(model.GEN_RENEWABLE, within=model.NODE)
        model.generationRenewableType = pyo.Param(model.GEN_RENEWABLE, within=model.GEN_RENEWABLE_TYPE)
        model.generationRenewableIec = pyo.Param(model.GEN_RENEWABLE, within=pyo.Integers)
        model.generationRenewableLcoe = pyo.Param(model.GEN_RENEWABLE, within=pyo.Any)
        model.generationRenewableProfile = pyo.Param(model.STAGE, model.GEN_RENEWABLE, model.TIME, within=pyo.Reals)
        model.generationRenewablePotentialMax = pyo.Param(model.STAGE, model.GEN_RENEWABLE, within=pyo.Reals)

        # model.generationRenewableOperationalInformation = pyo.Param(model.GEN_RENEWABLE, model.STAGE_OPERATIONAL, initialize=generationRenewableOperationalInformation_rule, within=pyo.Any)

        # - Conventional load
        model.convLoadNode = pyo.Param(model.LOAD, within=model.NODE)
        model.convLoadAnnualDemand = pyo.Param(model.LOAD, within=pyo.Reals)
        model.convLoadProfile = pyo.Param(model.STAGE, model.LOAD, model.TIME, within=pyo.Reals)

        # - Storage
        model.storageNode = pyo.Param(model.STORAGE, within=model.NODE)
        model.storageType = pyo.Param(model.STORAGE, within=model.STORAGE_TYPE)
        model.storageEtaOut = pyo.Param(model.STORAGE, within=pyo.PercentFraction)
        model.storageEtaIn = pyo.Param(model.STORAGE, within=pyo.PercentFraction)
        model.storageRatioVolume = pyo.Param(model.STORAGE, within=pyo.NonNegativeReals)
        model.storageSelfDischargeRate = pyo.Param(model.STORAGE, within=pyo.PercentFraction)
        model.storageDepthOfDischarge = pyo.Param(model.STORAGE, within=pyo.PercentFraction)
        model.storagePotentialMax = pyo.Param(model.STAGE, model.STORAGE, within=pyo.Reals)

        # - Transmission flows
        model.branchDistance = pyo.Param(model.BRANCH, within=pyo.NonNegativeReals)
        model.branchExistingCapacity = pyo.Param(model.BRANCH, within=pyo.NonNegativeReals)
        model.branchType = pyo.Param(model.BRANCH, within=pyo.Any)  # within=model.BRANCH_TYPE
        model.branchExistingExpand = pyo.Param(model.BRANCH, within=pyo.Boolean)

        # - Costs
        model.costSystemOperationEmissionPrice = pyo.Param(model.STAGE, within=pyo.NonNegativeReals)
        model.costSystemOperationFuel = pyo.Param(model.FUEL, within=pyo.NonNegativeReals)

        # --- Generation (thermal and renewable technologies)
        model.costGenerationCapex = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeReals)
        model.costGenerationDepreciationPeriod = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeIntegers)
        model.costGenerationInterestRate = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeReals)
        model.costGenerationOpexVariable = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeReals)
        model.costGenerationOpexFixed = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeReals)

        model.multiPeriodCostGenerationTotalInvestment = pyo.Param(model.STAGE, model.GEN_TYPE, within=pyo.NonNegativeReals)
        model.multiPeriodCostGenerationDecommissioning = pyo.Param(model.STAGE_DECOMMISSIONING, model.GEN_TYPE, within=pyo.NonNegativeReals)

        # --- Storage
        model.costStorageCapex = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeReals)
        model.costStorageDepreciationPeriod = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeIntegers)
        model.costStorageInterestRate = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeReals)
        model.costStorageOpexVariable = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeReals)
        model.costStorageOpexFixed = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeReals)

        model.multiPeriodCostStorageTotalInvestment = pyo.Param(model.STAGE, model.STORAGE_TYPE, within=pyo.NonNegativeReals)
        model.multiPeriodCostStorageDecommissioning = pyo.Param(model.STAGE_DECOMMISSIONING, model.STORAGE_TYPE, within=pyo.NonNegativeReals)

        # Decision variables ##############################################################################################################
        # - Bounds rules
        def _generatorThermalCapacity_bounds_rule(model, g, *stg):
            return (0, model.generationThermalPotentialMax[stg[1], g])

        def _generatorThermalNewCapacity_bounds_rule(model, g, stg):
            return (0, model.generationThermalPotentialMax[stg, g])

        def _generationThermal_bounds_rule(model, g, *stg):
            return (0, model.generationThermalPotentialMax[stg[1], g])

        def _generatorRenewableCapacity_bounds_rule(model, g, *stg):
            return (0, model.generationRenewablePotentialMax[stg[1], g])

        def _generatorRenewableNewCapacity_bounds_rule(model, g, stg):
            return (0, model.generationRenewablePotentialMax[stg, g])

        def _generationRenewable_bounds_rule(model, g, *stg):
            return (0, model.generationRenewablePotentialMax[stg[1], g])

        def _electricityStorageCapacity_bounds_rule(model, s, *stg):
            return (0, model.storagePotentialMax[stg[1], s])

        def _electricityStorageNewCapacity_bounds_rule(model, s, stg):
            return (0, model.storagePotentialMax[stg, s])

        def _electricityStorage_bounds_rule(model, g, *stg):
            return (0, model.storagePotentialMax[stg[1], g])

        # - generationThermal capacity: operational, investment, decommissioning
        model.generatorThermalCapacity = pyo.Var(model.GEN_THERMAL, model.STAGE_OPERATIONAL, bounds=_generatorThermalCapacity_bounds_rule, within=pyo.NonNegativeReals)
        model.generatorThermalNewCapacity = pyo.Var(model.GEN_THERMAL, model.STAGE, bounds=_generatorThermalNewCapacity_bounds_rule, within=pyo.NonNegativeReals)
        model.generatorThermalDecommissionedCapacity = pyo.Var(model.GEN_THERMAL, model.STAGE_DECOMMISSIONING, within=pyo.NonNegativeReals)

        # - generationRenewable capacity: operational, investment, decommissioning
        model.generatorRenewableCapacity = pyo.Var(model.GEN_RENEWABLE, model.STAGE_OPERATIONAL, bounds=_generatorRenewableCapacity_bounds_rule, within=pyo.NonNegativeReals)
        model.generatorRenewableNewCapacity = pyo.Var(model.GEN_RENEWABLE, model.STAGE, bounds=_generatorRenewableNewCapacity_bounds_rule, within=pyo.NonNegativeReals)
        model.generatorRenewableDecommissionedCapacity = pyo.Var(model.GEN_RENEWABLE, model.STAGE_DECOMMISSIONING, within=pyo.NonNegativeReals)

        # - Generator output (bounds set by constraint)
        model.generationThermal = pyo.Var(model.GEN_THERMAL, model.TIME, model.STAGE_OPERATIONAL, bounds=_generationThermal_bounds_rule, within=pyo.NonNegativeReals)
        model.generationRenewable = pyo.Var(model.GEN_RENEWABLE, model.TIME, model.STAGE_OPERATIONAL, bounds=_generationRenewable_bounds_rule, within=pyo.NonNegativeReals)
        model.curtailmentRenewable = pyo.Var(model.GEN_RENEWABLE, model.TIME, model.STAGE_OPERATIONAL, bounds=_generationRenewable_bounds_rule, within=pyo.NonNegativeReals)

        if not self.exclude_components["electricity_storage"]:
            # - Electricity storage capacity: operational, investment, decommissioning
            model.electricityStorageCapacity = pyo.Var(model.STORAGE, model.STAGE_OPERATIONAL, bounds=_electricityStorageCapacity_bounds_rule, within=pyo.NonNegativeReals)
            model.electricityStorageNewCapacity = pyo.Var(model.STORAGE, model.STAGE, bounds=_electricityStorageNewCapacity_bounds_rule, within=pyo.NonNegativeReals)
            model.electricityStorageDecommissionedCapacity = pyo.Var(model.STORAGE, model.STAGE_DECOMMISSIONING, within=pyo.NonNegativeReals)

            # - Electricity storage in- and output and level
            model.generationElectricityStorage = pyo.Var(model.STORAGE, model.TIME, model.STAGE_OPERATIONAL, bounds=_electricityStorage_bounds_rule, within=pyo.NonNegativeReals)
            model.consumptionElectricityStorage = pyo.Var(model.STORAGE, model.TIME, model.STAGE_OPERATIONAL, bounds=_electricityStorage_bounds_rule, within=pyo.NonNegativeReals)
            model.storageLevelElectricityStorage = pyo.Var(model.STORAGE, model.TIME, model.STAGE_OPERATIONAL, within=pyo.NonNegativeReals)

        def _flow_bounds_rule(model, n1, n2, t, stg):
            return (0, model.branchExistingCapacity[n1, n2])

        # - Transmission flows
        model.flow1 = pyo.Var(model.BRANCH, model.TIME, model.STAGE, bounds=_flow_bounds_rule, within=pyo.NonNegativeReals)
        model.flow2 = pyo.Var(model.BRANCH, model.TIME, model.STAGE, bounds=_flow_bounds_rule, within=pyo.NonNegativeReals)

        # Constraints #####################################################################################################################
        # - Thermal power generation, capacity expansion and decommissioning
        def maxGeneratorThermalOutput_rule(model, g, t, *stage):
            expr = model.generationThermal[g, t, stage] <= model.generatorThermalCapacity[g, stage]  # GW <= GW
            return expr

        model.c_maxGenThermal = pyo.Constraint(
            model.GEN_THERMAL,
            model.TIME,
            model.STAGE_OPERATIONAL,
            rule=maxGeneratorThermalOutput_rule,
        )

        def generationThermalCapacity_rule(model, g, *stage):
            if stage[0] == stage[1]:  # TODO: add pre-existing capacity parameter for initial stage, i.e. stage[0] == 1
                expr = model.generatorThermalCapacity[g, stage] == model.generatorThermalNewCapacity[g, stage[0]]
            else:
                expr = model.generatorThermalCapacity[g, stage] == model.generatorThermalCapacity[g, stage[0], stage[1] - 1] - model.generatorThermalDecommissionedCapacity[g, stage]
            return expr

        model.c_generationThermalCapacity = pyo.Constraint(model.GEN_THERMAL, model.STAGE_OPERATIONAL, rule=generationThermalCapacity_rule)

        def generationThermalCapacityDecommissioning_rule(model, g, *stage):
            expr = model.generatorThermalDecommissionedCapacity[g, stage] <= model.generatorThermalCapacity[g, stage[0], stage[1] - 1]
            return expr

        model.c_generationThermalCapacityDecommissioning = pyo.Constraint(model.GEN_THERMAL, model.STAGE_DECOMMISSIONING, rule=generationThermalCapacityDecommissioning_rule)

        def generationThermalCapacityMaximumPotential_rule(model, g, stage):
            expr = 0
            for stage_inv in range(stage):
                expr += model.generatorThermalCapacity[g, stage_inv + 1, stage]
            expr = expr <= model.generationThermalPotentialMax[stage, g]
            return expr

        model.c_generationThermalCapacityMaximumPotential = pyo.Constraint(model.GEN_THERMAL, model.STAGE, rule=generationThermalCapacityMaximumPotential_rule)

        # - Renewable power generation, capacity expansion and decommissioning
        def maxGeneratorRenewableOutput_rule(model, g, t, *stage):
            tmp_combined_profile = 0.0
            for stg in model.STAGE:
                tmp_combined_profile += model.operationalStageContributionGeneration[stage[0], stage[1], stg, model.generationRenewableType[g]] * model.generationRenewableProfile[stg, g, t]
            expr = model.generationRenewable[g, t, stage] + model.curtailmentRenewable[g, t, stage] == model.generatorRenewableCapacity[g, stage] * self.truncate(tmp_combined_profile, 6)  # GW <= GW
            # expr = model.generationRenewable[g, t, stage] * 10 ** 6 + model.curtailmentRenewable[g, t, stage] * 10 ** 6 == model.generatorRenewableCapacity[g, stage] * int(tmp_combined_profile * 10 ** 6)
            return expr

        model.c_maxGenRenewable = pyo.Constraint(
            model.GEN_RENEWABLE,
            model.TIME,
            model.STAGE_OPERATIONAL,
            rule=maxGeneratorRenewableOutput_rule,
        )

        def generationRenewableCapacity_rule(model, g, *stage):
            if stage[0] == stage[1]:  # TODO: add pre-existing capacity parameter for initial stage, i.e. stage[0] == 1
                expr = model.generatorRenewableCapacity[g, stage] == model.generatorRenewableNewCapacity[g, stage[0]]
            else:
                expr = model.generatorRenewableCapacity[g, stage] == model.generatorRenewableCapacity[g, stage[0], stage[1] - 1] - model.generatorRenewableDecommissionedCapacity[g, stage]
            return expr

        model.c_generationRenewableCapacity = pyo.Constraint(model.GEN_RENEWABLE, model.STAGE_OPERATIONAL, rule=generationRenewableCapacity_rule)

        def generationRenewableCapacityDecommissioning_rule(model, g, *stage):
            expr = model.generatorRenewableDecommissionedCapacity[g, stage] <= model.generatorRenewableCapacity[g, stage[0], stage[1] - 1]
            return expr

        model.c_generationRenewableCapacityDecommissioning = pyo.Constraint(model.GEN_RENEWABLE, model.STAGE_DECOMMISSIONING, rule=generationRenewableCapacityDecommissioning_rule)

        def generationRenewableCapacityMaximumPotential_rule(model, g, stage):
            expr = 0
            for stage_inv in range(stage):
                expr += model.generatorRenewableCapacity[g, stage_inv + 1, stage]
            expr = expr <= model.generationRenewablePotentialMax[stage, g]
            return expr

        model.c_generationRenewableCapacityMaximumPotential = pyo.Constraint(model.GEN_RENEWABLE, model.STAGE, rule=generationRenewableCapacityMaximumPotential_rule)

        # - Electricity storage generation, consumption, capacity expansion and decommissioning
        if not self.exclude_components["electricity_storage"]:

            def electricityStorageCapacity_rule(model, s, *stage):
                if stage[0] == stage[1]:  # TODO: add pre-existing capacity parameter for initial stage, i.e. stage[0] == 1
                    expr = model.electricityStorageCapacity[s, stage] == model.electricityStorageNewCapacity[s, stage[0]]
                else:
                    expr = model.electricityStorageCapacity[s, stage] == model.electricityStorageCapacity[s, stage[0], stage[1] - 1] - model.electricityStorageDecommissionedCapacity[s, stage]
                return expr

            model.c_electricityStorageCapacity = pyo.Constraint(model.STORAGE, model.STAGE_OPERATIONAL, rule=electricityStorageCapacity_rule)

            def electricityStorageCapacityDecommissioning_rule(model, s, *stage):
                expr = model.electricityStorageDecommissionedCapacity[s, stage] <= model.electricityStorageCapacity[s, stage[0], stage[1] - 1]
                return expr

            model.c_electricityStorageCapacityDecommissioning = pyo.Constraint(model.STORAGE, model.STAGE_DECOMMISSIONING, rule=electricityStorageCapacityDecommissioning_rule)

            def electricityStorageCapacityMaximumPotential_rule(model, s, stage):
                expr = 0
                for stage_inv in range(stage):
                    expr += model.electricityStorageCapacity[s, stage_inv + 1, stage]
                expr = expr <= model.storagePotentialMax[stage, s]
                return expr

            model.c_electricityStorageCapacityMaximumPotential = pyo.Constraint(model.STORAGE, model.STAGE, rule=electricityStorageCapacityMaximumPotential_rule)

            def maxElectricityStorageOutput_rule(model, s, t, *stage):
                expr = model.generationElectricityStorage[s, t, stage] <= model.electricityStorageCapacity[s, stage]  # GW <= GW
                return expr

            model.c_maxElectricityStorageOutput = pyo.Constraint(
                model.STORAGE,
                model.TIME,
                model.STAGE_OPERATIONAL,
                rule=maxElectricityStorageOutput_rule,
            )

            def maxElectricityStorageInput_rule(model, s, t, *stage):
                expr = model.consumptionElectricityStorage[s, t, stage] <= model.electricityStorageCapacity[s, stage]  # GW <= GW
                return expr

            model.c_maxElectricityStorageInput = pyo.Constraint(
                model.STORAGE,
                model.TIME,
                model.STAGE_OPERATIONAL,
                rule=maxElectricityStorageInput_rule,
            )

            def maxElectricityStorageLevel_rule(model, s, t, *stage):
                expr = model.storageLevelElectricityStorage[s, t, stage] <= model.electricityStorageCapacity[s, stage] * model.storageRatioVolume[s]  # GWh <= GWh
                return expr

            model.c_maxElectricityStorageLevel = pyo.Constraint(
                model.STORAGE,
                model.TIME,
                model.STAGE_OPERATIONAL,
                rule=maxElectricityStorageLevel_rule,
            )

            def electricityStorageContinuity_rule(model, s, t, *stage):
                expr = 0
                if t < max(model.TIME):  # (1 - model.storageSelfDischargeRate[s]) *
                    expr = model.storageLevelElectricityStorage[s, t + 1, stage] == model.storageLevelElectricityStorage[s, t, stage] + model.storageEtaIn[s] * model.consumptionElectricityStorage[
                        s, t, stage
                    ] - model.generationElectricityStorage[s, t, stage] * self.truncate(
                        1 / model.storageEtaOut[s], 6
                    )  # Storage continuity equation
                else:
                    expr = (
                        model.storageLevelElectricityStorage[s, t, stage]
                        + model.storageEtaIn[s] * model.consumptionElectricityStorage[s, t, stage]
                        - model.generationElectricityStorage[s, t, stage] * self.truncate(1 / model.storageEtaOut[s], 6)
                        >= model.storageLevelElectricityStorage[s, min(model.TIME), stage]
                    )  # Final equal to initial storage level of a representative period or the full year
                return expr

            model.c_electricityStorageContinuity = pyo.Constraint(
                model.STORAGE,
                model.TIME,
                model.STAGE_OPERATIONAL,
                rule=electricityStorageContinuity_rule,
            )

        # - Transmission flows
        def maxTransmissionLimit_rule(model, n1, n2, t, stage):
            expr = model.flow1[n1, n2, t, stage] + model.flow2[n1, n2, t, stage] <= model.branchExistingCapacity[n1, n2]
            return expr

        model.c_maxTransmissionLimit = pyo.Constraint(model.BRANCH, model.TIME, model.STAGE, rule=maxTransmissionLimit_rule)

        # - Nodal power balance (market clearing constraint)
        def nodalPowerBalance_rule(model, n, t, stage):
            expr = 0

            # Thermal generation
            for g in model.GEN_THERMAL:
                if model.generationThermalNode[g] == n:
                    for s in range(1, stage + 1):
                        expr += model.generationThermal[g, t, s, stage]

            # Renewable generation
            for g in model.GEN_RENEWABLE:
                if model.generationRenewableNode[g] == n:
                    for s in range(1, stage + 1):
                        expr += model.generationRenewable[g, t, s, stage]

            # Conventional consumer load
            for c in model.LOAD:
                if model.convLoadNode[c] == n:
                    if model.convLoadAnnualDemand[c] != -1:
                        # expr += -model.convLoadAnnualDemand[c] * model.convLoadProfile[c, t] / sum(model.convLoadProfile[c, t1] for t1 in model.TIME)
                        expr -= self.truncate(model.convLoadProfile[stage, c, t] / 1000.0 * (1.0 + 0.05 * stage), 6)  # GW
                    else:
                        expr -= self.truncate(model.convLoadProfile[stage, c, t] / 1000.0 * (1.0 + 0.05 * stage), 6)  # GW

            # Electricity storage
            if not self.exclude_components["electricity_storage"]:
                for su in model.STORAGE:
                    if model.storageNode[su] == n:
                        for s in range(1, stage + 1):
                            expr += model.generationElectricityStorage[su, t, s, stage]
                            expr -= model.consumptionElectricityStorage[su, t, s, stage]

            expr += sum(model.flow1[i, n, t, stage] for i in model.NODE_IN[n])
            expr += -sum(model.flow2[i, n, t, stage] for i in model.NODE_IN[n]) * self.truncate(1 / 0.97, 6)  # TODO: Include transmission loss factors as parameters
            expr += -sum(model.flow1[n, j, t, stage] for j in model.NODE_OUT[n]) * self.truncate(1 / 0.97, 6)  # TODO: Include transmission loss factors as parameters
            expr += sum(model.flow2[n, j, t, stage] for j in model.NODE_OUT[n])

            expr = expr == 0

            if (type(expr) is bool) and (expr == True):
                # Trivial constraint
                expr = pyo.Constraint.Skip
            return expr

        model.c_nodalPowerBalance = pyo.Constraint(model.NODE, model.TIME, model.STAGE, rule=nodalPowerBalance_rule)

        # Objective #######################################################################################################################
        model.costInvestment = pyo.Var(model.STAGE_MODEL, within=pyo.NonNegativeReals)
        model.costSystemOperation = pyo.Var(model.STAGE_MODEL, within=pyo.NonNegativeReals)

        def costInvestment_rule(model, stage):
            """CAPEX and OPEX for investment decisions, ...(NPV)"""
            if stage <= model.numberOfStages:
                expr = self.getCostInvestments(model, stage)
            else:
                expr = 0
            return model.costInvestment[stage] == expr

        model.cCostInvestment = pyo.Constraint(model.STAGE_MODEL, rule=costInvestment_rule)

        def costSystemOperation_rule(model, stage):
            """System operation costs for thermal generationThermal, ... (NPV)"""
            if stage > 1:
                expr = self.getCostSystemOperation(model, stage - 1)
            else:
                expr = 0
            return model.costSystemOperation[stage] == expr

        model.cCostSystemOperation = pyo.Constraint(model.STAGE_MODEL, rule=costSystemOperation_rule)

        def costTotal_rule(model, stage):
            """Total operation costs: cost of thermal generationThermal, ... (NPV)"""
            expr = model.costInvestment[stage] + model.costSystemOperation[stage]
            return expr

        model.costTotal = pyo.Expression(model.STAGE_MODEL, rule=costTotal_rule)

        def costTotal_Objective_rule(model):
            expr = pyo.summation(model.costTotal)
            return expr

        model.obj = pyo.Objective(rule=costTotal_Objective_rule, sense=pyo.minimize)

        return model

    def getCostInvestments(self, model, stage, includeRelativeOpex=False, subtractSalvage=True):
        """Investment cost, including lifetime OPEX (NPV)"""
        cost_investment = 0.0

        for g in model.GEN_THERMAL:
            cost_investment += self.getCostInvestmentByUnitGroup(model, g, stage, "generationThermal")  # unit_group = "generationThermal"

        for g in model.GEN_RENEWABLE:
            cost_investment += self.getCostInvestmentByUnitGroup(model, g, stage, "generationRenewable")  # unit_group = "generationRenewable"

        if not self.exclude_components["electricity_storage"]:
            for s in model.STORAGE:
                cost_investment += self.getCostInvestmentByUnitGroup(model, s, stage, "electricityStorage")  # unit_group = "electricityStorage"

        return cost_investment

    def getCostInvestmentByUnitGroup(self, model, u, stage, unit_group=None):
        """Expression for investment cost of specified unit group"""
        cost_investment = 0.0
        if unit_group == "generationThermal":
            gen_type = model.generationThermalType[u]
            cost_investment += model.multiPeriodCostGenerationTotalInvestment[stage, gen_type] * model.generatorThermalNewCapacity[u, stage]
            for stg_dec in range(stage + 1, model.numberOfStages + 1):  # e.g. investment in 'stage' = 1, corresponding decommissioning in stages [2, 3, 4]
                cost_investment -= model.multiPeriodCostGenerationDecommissioning[stage, stg_dec, gen_type] * model.generatorThermalDecommissionedCapacity[u, stage, stg_dec]
        elif unit_group == "generationRenewable":
            gen_type = model.generationRenewableType[u]
            cost_investment += model.multiPeriodCostGenerationTotalInvestment[stage, gen_type] * model.generatorRenewableNewCapacity[u, stage]
            for stg_dec in range(stage + 1, model.numberOfStages + 1):  # e.g. investment in 'stage' = 1, corresponding decommissioning in stages [2, 3, 4]
                cost_investment -= model.multiPeriodCostGenerationDecommissioning[stage, stg_dec, gen_type] * model.generatorRenewableDecommissionedCapacity[u, stage, stg_dec]
        elif unit_group == "electricityStorage" and not self.exclude_components["electricity_storage"]:
            storage_type = model.storageType[u]
            cost_investment += model.multiPeriodCostStorageTotalInvestment[stage, storage_type] * model.electricityStorageNewCapacity[u, stage]
            for stg_dec in range(stage + 1, model.numberOfStages + 1):  # e.g. investment in 'stage' = 1, corresponding decommissioning in stages [2, 3, 4]
                cost_investment -= model.multiPeriodCostStorageDecommissioning[stage, stg_dec, storage_type] * model.electricityStorageDecommissionedCapacity[u, stage, stg_dec]
        else:
            print("\n!!! ERROR WRONG OR NO UNIT GROUP SPECIFIED WHEN OBTAINING INVESTMENT COST !!!\n")

        return cost_investment

    def getCostSystemOperation(self, model, stage):
        """Operational costs: cost of gen, load shed (NPV)"""
        cost_system_operation = 0.0

        (discount_factor_sum, discount_factor_perpetuity) = self.getDiscountFactorSum(pyo.value(model.financePresentValueInterestRate), stage, model.yearsPerStage)

        if stage == pyo.value(model.numberOfStages):
            discount_factor_sum = discount_factor_sum + discount_factor_perpetuity
        else:
            del discount_factor_perpetuity
            # print(discount_factor_sum)
        # print("Stage: " + str(stage), "Discount factor sum: " + str(discount_factor_sum), "Period weight factor: " + str(pyo.value(model.periodWeightFactor)))

        for stage_operational_idx in model.STAGE_OPERATIONAL:
            if stage_operational_idx[1] == stage:
                for t in model.TIME:
                    # Thermal generation units

                    # TODO: Thermal efficiencies
                    # for g in model.GEN_THERMAL
                    #     operational_stage = self.getMultiPeriodInformation(
                    #         None,
                    #         None,
                    #         None,
                    #         model.financePresentValueInterestRate,
                    #         [model.costGenerationDepreciationPeriod[x, model.generationThermalType[g]] for x in model.STAGE],
                    #         stage[0],
                    #         max(model.STAGE),
                    #         model.yearsPerStage,
                    #         False,
                    #     )[
                    #         2
                    #     ]  # Only third output relevant
                    #     tmp_combined_efficiency = np.multiply(
                    #         operational_stage[stage[0] - 1, :], np.asarray([model.generationRenewableProfile[s + 1, g, t] for s in range(operational_stage.shape[1])])
                    #     ).sum()  # Weighted sum of renewable availability = sum(Investment period weigths * renewable availability profile)

                    cost_system_operation += pyo.quicksum(
                        model.generationThermal[g, t, stage_operational_idx]
                        * model.periodWeightFactor
                        / model.generationThermalEta[g]
                        * (model.costSystemOperationFuel[model.generationThermalFuel[g]] + model.emissionFactor[model.generationThermalFuel[g]] * model.costSystemOperationEmissionPrice[stage])
                        * discount_factor_sum
                        for g in model.GEN_THERMAL
                    )

                    if t == 1:
                        print(
                            "thermal sysop",
                            0,
                            model.periodWeightFactor
                            / model.generationThermalEta[0]
                            * (model.costSystemOperationFuel[model.generationThermalFuel[0]] + model.emissionFactor[model.generationThermalFuel[0]] * model.costSystemOperationEmissionPrice[stage])
                            * discount_factor_sum,
                        )

                    # Renewable generation units
                    cost_system_operation += pyo.quicksum(
                        model.generationRenewable[g, t, stage_operational_idx] * model.costGenerationOpexVariable[stage, model.generationRenewableType[g]] * model.periodWeightFactor * discount_factor_sum
                        for g in model.GEN_RENEWABLE
                    )

                    if t == 1:
                        print("renewable sysop", 0, model.costGenerationOpexVariable[stage, model.generationRenewableType[0]] * model.periodWeightFactor * discount_factor_sum)

                    # Electricity storage units
                    if not self.exclude_components["electricity_storage"]:
                        cost_system_operation += pyo.quicksum(
                            (model.generationElectricityStorage[s, t, stage_operational_idx] + model.consumptionElectricityStorage[s, t, stage_operational_idx])
                            * model.costStorageOpexVariable[stage, model.storageType[s]]
                            * model.periodWeightFactor
                            * discount_factor_sum
                            for s in model.STORAGE
                        )

        return cost_system_operation

    def createConcreteModel(self, dict_data):
        """Create concrete Pyomo model for the EMPRISE optimization problem instance

        Parameters
        ----------
        dict_data : dictionary
            dictionary containing the model data. This can be created with
            the createModelData(...) method

        Returns
        -------
            Concrete pyomo model
        """

        concrete_model = self.abstract_model.create_instance(data=dict_data, name="EMPRISE Model", namespace="emprise", report_timing=False)
        return concrete_model

    def createModelData(self, input_data):
        """Create model data in dictionary format

        Parameters
        ----------
        input_data : emprise.InputData object
            containing structural and timeseries input data

        Returns
        -------
        dictionary with pyomo data (in pyomo format)
        """

        di = {}

        # --- Sets
        di["NODE"] = {None: input_data.node["id"].tolist()}
        di["FUEL"] = {None: sorted(set(input_data.cost["systemOperation"]["fuel"].keys()))}
        di["AREA"] = {None: input_data.getAllAreas()}
        di["TIME"] = {None: input_data.time_range}
        di["BRANCH"] = {None: [(row["node_from"], row["node_to"]) for k, row in input_data.branch.iterrows() if row["node_from"] in di["NODE"][None] and row["node_to"] in di["NODE"][None]]}

        # --- General
        di["numberOfStages"] = {None: input_data.number_of_stages}  # -
        di["willingnessToPay"] = {None: input_data.general["willingnessToPay"]}  # EUR/MWh
        di["financePresentValueInterestRate"] = {None: input_data.general["finance"]["interestRate"]}  # -

        di["yearsPerStage"] = {None: input_data.general["yearsPerStage"]}  # -
        di["periodWeightFactor"] = {None: input_data.ts_periodWeightFactor}  # -
        di["emissionFactor"] = input_data.const_emission_factor  # tCO2eq/MWh_th

        # --- Generation thermal
        di["generationThermalNode"] = {}
        di["generationThermalType"] = {}
        di["generationThermalFuel"] = {}
        di["generationThermalEta"] = {}
        di["generationThermalPotentialMax"] = {}
        k = 0
        for it, row in input_data.generator_thermal.iterrows():
            if row["node"] in di["NODE"][None]:
                di["generationThermalNode"][k] = row["node"]
                di["generationThermalType"][k] = row["type"]
                di["generationThermalFuel"][k] = row["fuel"]
                di["generationThermalEta"][k] = row["eta"]
                for stg in range(1, input_data.number_of_stages + 1):
                    di["generationThermalPotentialMax"][(stg, k)] = float(row["potential_max"].split("::")[stg - 1])
                k = k + 1
        di["GEN_THERMAL"] = {None: range(k)}
        di["GEN_THERMAL_TYPE"] = {None: sorted(set(di["generationThermalType"].values()))}
        del k
        # di["financeDeprecationPeriodGenerationThermal"] = {
        #     k: input_data.general["finance"]["deprecationPeriod"]["generationThermal"][k] for k in input_data.general["finance"]["deprecationPeriod"]["generationThermal"].keys() & di["GEN_THERMAL_TYPE"][None]
        # }

        # --- Generation renewable
        di["generationRenewableNode"] = {}
        di["generationRenewableType"] = {}
        di["generationRenewableIec"] = {}
        di["generationRenewableLcoe"] = {}
        di["generationRenewablePotentialMax"] = {}
        di["generationRenewableProfile"] = {}
        k = 0
        for it, row in input_data.generator_renewable.iterrows():
            if row["node"] in di["NODE"][None]:
                di["generationRenewableNode"][k] = row["node"]
                di["generationRenewableType"][k] = row["type"]
                di["generationRenewableIec"][k] = row["iec"]
                di["generationRenewableLcoe"][k] = "LCOE_" + str(row["lcoe"])
                ref = row["node"]
                for stg in range(1, input_data.number_of_stages + 1):
                    di["generationRenewablePotentialMax"][(stg, k)] = float(row["potential_max"].split("::")[stg - 1])
                    # print(stg, stg - 1)
                    for i, t in enumerate(input_data.time_range):
                        if row["type"].startswith("SOLAR"):
                            tmp_tuple = (
                                row["node"],
                                row["type"],
                                "LCOE_" + str(row["lcoe"]),
                                str(input_data.reference_years[stg - 1]),
                            )
                            di["generationRenewableProfile"][(stg, k, t)] = input_data.ts_generator_solar[tmp_tuple][i]
                            del tmp_tuple
                        elif row["type"].startswith("ONSHORE") or row["type"].startswith("OFFSHORE"):
                            tmp_tuple = (
                                row["node"],
                                row["type"],
                                "LCOE_" + str(row["lcoe"]),
                                str(input_data.reference_years[stg - 1]),
                            )
                            di["generationRenewableProfile"][(stg, k, t)] = input_data.ts_generator_wind[tmp_tuple][i]
                            del tmp_tuple
                        else:
                            print(
                                "\n\n !!! WARNING: no timeseries data found for type {} !!!\n\n",
                                row["type"],
                            )
                k = k + 1
        di["GEN_RENEWABLE"] = {None: range(k)}
        di["GEN_RENEWABLE_TYPE"] = {None: sorted(set(di["generationRenewableType"].values()))}
        del k
        # di["financeDeprecationPeriodGenerationRenewable"] = {
        #     k: input_data.general["finance"]["deprecationPeriod"]["generationRenewable"][k] for k in input_data.general["finance"]["deprecationPeriod"]["generationRenewable"].keys() & di["GEN_RENEWABLE_TYPE"][None]
        # }

        # --- Conventional load
        di["convLoadAnnualDemand"] = {}
        di["convLoadProfile"] = {}
        di["convLoadNode"] = {}
        k = 0
        for _, row in input_data.consumer_conventional.iterrows():
            if row["node"] in di["NODE"][None]:
                di["convLoadNode"][k] = row["node"]
                di["convLoadAnnualDemand"][k] = row["annualDemand"]
                ref = row["node"]
                for stg in range(1, input_data.number_of_stages + 1):
                    for i, t in enumerate(input_data.time_range):
                        di["convLoadProfile"][(stg, k, t)] = input_data.ts_consumer_conventional[(ref, str(input_data.reference_years[stg - 1]))][i]
                k = k + 1
        di["LOAD"] = {None: range(k)}
        del k

        # --- Storage
        di["storageNode"] = {}
        di["storageType"] = {}
        di["storageEtaOut"] = {}
        di["storageEtaIn"] = {}
        di["storageRatioVolume"] = {}
        di["storageSelfDischargeRate"] = {}
        di["storageDepthOfDischarge"] = {}
        di["storagePotentialMax"] = {}
        k = 0
        for it, row in input_data.storage.iterrows():
            if row["node"] in di["NODE"][None] and row["include"]:
                di["storageNode"][k] = row["node"]
                di["storageType"][k] = row["type"]
                di["storageEtaOut"][k] = row["eta_out"]
                di["storageEtaIn"][k] = row["eta_in"]
                di["storageRatioVolume"][k] = row["ratio_volume"]
                di["storageSelfDischargeRate"][k] = row["sdr"]
                di["storageDepthOfDischarge"][k] = row["dod"]
                for stg in range(1, input_data.number_of_stages + 1):
                    di["storagePotentialMax"][(stg, k)] = float(row["potential_max"].split("::")[stg - 1])
                k = k + 1
        if k == 0:
            di["STORAGE"] = {None: pyo.Set.Skip}
            di["STORAGE_TYPE"] = {None: []}
        else:
            di["STORAGE"] = {None: range(k)}
            di["STORAGE_TYPE"] = {None: sorted(set(di["storageType"].values()))}
        del k

        # --- Cross-border exchange
        di["branchDistance"] = {}
        di["branchType"] = {}
        di["branchExistingCapacity"] = {}
        di["branchExistingExpand"] = {}
        for k, row in input_data.branch.iterrows():
            if row["node_from"] in di["NODE"][None] and row["node_to"] in di["NODE"][None]:
                di["branchDistance"][(row["node_from"], row["node_to"])] = row["distance"]
                di["branchType"][(row["node_from"], row["node_to"])] = row["type"]
                di["branchExistingCapacity"][(row["node_from"], row["node_to"])] = row["capacity"]
                di["branchExistingExpand"][(row["node_from"], row["node_to"])] = row["expand"]

        # --- System operation costs
        di["costSystemOperationFuel"] = input_data.cost["systemOperation"]["fuel"]  # EUR/MWh
        di["costSystemOperationEmissionPrice"] = input_data.cost["systemOperation"]["emissionPrice"]  # EUR/tCO2eq

        # --- Investment costs
        # di["costInvestmentThermalCapex"] = {k: input_data.cost["investment"]["capexThermal"][k] for k in input_data.cost["investment"]["capexThermal"].keys() & di["GEN_THERMAL_TYPE"][None]}  # EUR/MW
        # di["costInvestmentThermalOpex"] = {k: input_data.cost["investment"]["opexThermal"][k] for k in input_data.cost["investment"]["opexThermal"].keys() & di["GEN_THERMAL_TYPE"][None]}  # EUR/MW/a
        # di["costInvestmentRenewableCapex"] = {k: input_data.cost["investment"]["capexRenewable"][k] for k in input_data.cost["investment"]["capexRenewable"].keys() & di["GEN_RENEWABLE_TYPE"][None]}  # EUR/MW
        # di["costInvestmentRenewableOpex"] = {k: input_data.cost["investment"]["opexRenewable"][k] for k in input_data.cost["investment"]["opexRenewable"].keys() & di["GEN_RENEWABLE_TYPE"][None]}  # EUR/MW/a

        di["costGenerationCapex"] = {}
        di["costGenerationDepreciationPeriod"] = {}
        di["costGenerationInterestRate"] = {}
        di["costGenerationOpexVariable"] = {}
        di["costGenerationOpexFixed"] = {}

        di["costStorageCapex"] = {}
        di["costStorageDepreciationPeriod"] = {}
        di["costStorageInterestRate"] = {}
        di["costStorageOpexVariable"] = {}
        di["costStorageOpexFixed"] = {}

        for stg in range(1, input_data.number_of_stages + 1):
            for k, row in input_data.cost_generation.iterrows():
                gen_type = row["type"]
                if gen_type in di["GEN_THERMAL_TYPE"][None] + di["GEN_RENEWABLE_TYPE"][None]:
                    di["costGenerationCapex"][(stg, gen_type)] = float(row["capex"].split("::")[stg - 1])  # EUR/MW
                    di["costGenerationDepreciationPeriod"][(stg, gen_type)] = int(row["depreciation"].split("::")[stg - 1])  # a
                    di["costGenerationInterestRate"][(stg, gen_type)] = float(row["interest_rate"].split("::")[stg - 1])  # p.u.
                    di["costGenerationOpexVariable"][(stg, gen_type)] = float(row["opex_variable"].split("::")[stg - 1])  # EUR/MWh
                    di["costGenerationOpexFixed"][(stg, gen_type)] = float(row["opex_fixed"].split("::")[stg - 1])  # EUR/MW/a
            for k, row in input_data.cost_storage.iterrows():
                storage_type = row["type"]
                if storage_type in di["STORAGE_TYPE"][None]:
                    di["costStorageCapex"][(stg, storage_type)] = float(row["capex"].split("::")[stg - 1])  # EUR/MW
                    di["costStorageDepreciationPeriod"][(stg, storage_type)] = int(row["depreciation"].split("::")[stg - 1])  # a
                    di["costStorageInterestRate"][(stg, storage_type)] = float(row["interest_rate"].split("::")[stg - 1])  # p.u.
                    di["costStorageOpexVariable"][(stg, storage_type)] = float(row["opex_variable"].split("::")[stg - 1])  # EUR/MWh
                    di["costStorageOpexFixed"][(stg, storage_type)] = float(row["opex_fixed"].split("::")[stg - 1])  # EUR/MW/a

        tmp_stage = range(1, 1 + di["numberOfStages"][None])
        tmp_stage_operational = [(x, y) for x in tmp_stage for y in tmp_stage if x <= y]
        # tmp_stage_decommissioning = [(x, y) for x in tmp_stage for y in tmp_stage if x < y]

        di["operationalStageContributionGeneration"] = {}
        di["multiPeriodCostGenerationTotalInvestment"] = {}
        di["multiPeriodCostGenerationDecommissioning"] = {}

        di["operationalStageContributionStorage"] = {}
        di["multiPeriodCostStorageTotalInvestment"] = {}
        di["multiPeriodCostStorageDecommissioning"] = {}
        for stg_op in tmp_stage_operational:
            for gen_type in di["GEN_THERMAL_TYPE"][None] + di["GEN_RENEWABLE_TYPE"][None]:
                (total_cost, decommissioning_redemption, operational_stage_contribution_matrix) = self.getMultiPeriodInformation(
                    [di["costGenerationCapex"][(x, gen_type)] for x in tmp_stage],
                    [di["costGenerationOpexFixed"][(x, gen_type)] for x in tmp_stage],
                    [di["costGenerationInterestRate"][(x, gen_type)] for x in tmp_stage],
                    di["financePresentValueInterestRate"][None],
                    [di["costGenerationDepreciationPeriod"][(x, gen_type)] for x in tmp_stage],
                    stg_op[0],
                    di["numberOfStages"][None],
                    di["yearsPerStage"][None],
                    False,
                )

                # print(stg_op, gen_type)
                # print(total_cost, decommissioning_redemption, operational_stage_contribution_matrix)

                if stg_op[0] == stg_op[1]:
                    di["multiPeriodCostGenerationTotalInvestment"][(stg_op[0], gen_type)] = total_cost
                else:
                    di["multiPeriodCostGenerationDecommissioning"][(stg_op[0], stg_op[1], gen_type)] = decommissioning_redemption[stg_op[1] - stg_op[0] - 1]

                for stg in tmp_stage:
                    # print((stg_op[0], stg_op[1], stg, gen_type), operational_stage_contribution_matrix[stg_op[1] - 1, stg - 1])
                    di["operationalStageContributionGeneration"][(stg_op[0], stg_op[1], stg, gen_type)] = operational_stage_contribution_matrix[stg_op[1] - 1, stg - 1]

            for storage_type in di["STORAGE_TYPE"][None]:
                (total_cost, decommissioning_redemption, operational_stage_contribution_matrix) = self.getMultiPeriodInformation(
                    [di["costStorageCapex"][(x, storage_type)] for x in tmp_stage],
                    [di["costStorageOpexFixed"][(x, storage_type)] for x in tmp_stage],
                    [di["costStorageInterestRate"][(x, storage_type)] for x in tmp_stage],
                    di["financePresentValueInterestRate"][None],
                    [di["costStorageDepreciationPeriod"][(x, storage_type)] for x in tmp_stage],
                    stg_op[0],
                    di["numberOfStages"][None],
                    di["yearsPerStage"][None],
                    False,
                )

                if stg_op[0] == stg_op[1]:
                    di["multiPeriodCostStorageTotalInvestment"][(stg_op[0], storage_type)] = total_cost
                else:
                    di["multiPeriodCostStorageDecommissioning"][(stg_op[0], stg_op[1], storage_type)] = decommissioning_redemption[stg_op[1] - stg_op[0] - 1]

                for stg in tmp_stage:
                    di["operationalStageContributionStorage"][(stg_op[0], stg_op[1], stg, storage_type)] = operational_stage_contribution_matrix[stg_op[1] - 1, stg - 1]

        return {"emprise": di}

    def updateOperationalScenarioInformation(self, dict_data, input_data, sysop_sc, reference_years):
        for k, row in input_data.consumer_conventional.iterrows():
            # dict_data['emprise']['convLoadNode'][k] = row['node']
            # dict_data['emprise']['convLoadAnnualDemand'][k] = row['annualDemand']
            # ref = row['node']
            for stg in range(1, len(sysop_sc)):
                for i, t in enumerate(input_data.time_range):
                    if row["node"] in dict_data["emprise"]["NODE"][None]:
                        dict_data["emprise"]["convLoadProfile"][(stg, k, t)] = input_data.ts_consumer_conventional[(row["node"], str(reference_years[sysop_sc[stg] - 1]))][i]

        for k, row in input_data.generator_renewable.iterrows():
            for stg in range(1, len(sysop_sc)):
                for i, t in enumerate(input_data.time_range):
                    if row["node"] in dict_data["emprise"]["NODE"][None]:
                        if row["type"].startswith("SOLAR"):
                            tmp_tuple = (
                                row["node"],
                                row["type"],
                                "LCOE_" + str(row["lcoe"]),
                                str(reference_years[sysop_sc[stg] - 1]),
                            )
                            dict_data["emprise"]["generationRenewableProfile"][(stg, k, t)] = input_data.ts_generator_solar[tmp_tuple][i]
                            del tmp_tuple
                        elif row["type"].startswith("ONSHORE") or row["type"].startswith("OFFSHORE"):
                            tmp_tuple = (
                                row["node"],
                                row["type"],
                                "LCOE_" + str(row["lcoe"]),
                                str(reference_years[sysop_sc[stg] - 1]),
                            )
                            dict_data["emprise"]["generationRenewableProfile"][(stg, k, t)] = input_data.ts_generator_wind[tmp_tuple][i]
                            del tmp_tuple
                        else:
                            print(
                                "\n\n !!! WARNING: no timeseries data found for type {} !!!\n\n",
                                row["type"],
                            )

        return dict_data

    def createScenarioTreeGraph(
        self,
        n_stages,
        n_inv_sc,
        n_sysop_sc,
        scenario_probability=None,
        path_result_dir="",
    ):
        """Generate model instance with data. Alternative to .dat files

        Parameters
        ----------
        n_inv_sc : int
            number of investment scenarios.
        n_sysop_sc: int
            number of investment scenarios.
        scenario_probability : work in progress

        Returns
        -------
        PySP multi-stage scenario tree model

        """

        if scenario_probability is None:
            #   equal probability:
            scenario_probability = 1 / n_sysop_sc / n_inv_sc

        G = networkx.DiGraph()

        # Multi-stage scenario tree
        # Example for 3 planning stages (e.g. 2020, 2030, 2040)
        # and 2 strategic uncertainty scenario variants (e.g. high and low investment cost)
        # and 2 system operation uncertainty scenarios (e.g. two meteorological reference years) throughout the scenario tree
        # stage:            1       2       3       4
        # inv decisions:    yes     yes     yes     no
        # sysop decisions:  no      yes     yes     yes
        # #nodes:           1       4       16      32

        for stage in range(1, n_stages + 2):
            if stage == n_stages + 1:  # Last stage with nodes only representing system operation decisions
                node_range = range(1, (n_inv_sc * n_sysop_sc) ** (stage - 2) * n_sysop_sc + 1)
            else:
                node_range = range(1, (n_inv_sc * n_sysop_sc) ** (stage - 1) + 1)

            for node in node_range:

                scenario_node_name = "sc_{}_{}".format(stage, node)
                cost_var_name = "costTotal[{}]".format(stage)
                dec_var_names = []
                der_dec_var_names = [
                    "costInvestment[{}]".format(stage),
                    "costSystemOperation[{}]".format(stage),
                ]

                if stage <= n_stages:
                    dec_var_names.append("generatorThermalNewCapacity[*,{}]".format(stage))
                    dec_var_names.append("generatorThermalCapacity[*,*,{}]".format(stage))

                    dec_var_names.append("generatorRenewableNewCapacity[*,{}]".format(stage))
                    dec_var_names.append("generatorRenewableCapacity[*,*,{}]".format(stage))

                    if not self.exclude_components["electricity_storage"]:
                        dec_var_names.append("electricityStorageNewCapacity[*,{}]".format(stage))
                        dec_var_names.append("electricityStorageCapacity[*,*,{}]".format(stage))

                    if stage > 1:
                        dec_var_names.append("generatorRenewableDecommissionedCapacity[*,*,{}]".format(stage))
                        dec_var_names.append("generatorThermalDecommissionedCapacity[*,*,{}]".format(stage))

                        if not self.exclude_components["electricity_storage"]:
                            dec_var_names.append("electricityStorageDecommissionedCapacity[*,*,{}]".format(stage))

                if stage > 1:  # first stage only has investment decisions
                    # print(stage, "generationThermal[*,*,*,{}]".format(stage - 1))
                    der_dec_var_names.append("generationThermal[*,*,*,{}]".format(stage - 1))

                    der_dec_var_names.append("generationRenewable[*,*,*,{}]".format(stage - 1))
                    der_dec_var_names.append("curtailmentRenewable[*,*,*,{}]".format(stage - 1))

                    if not self.exclude_components["electricity_storage"]:
                        der_dec_var_names.append("generationElectricityStorage[*,*,*,{}]".format(stage - 1))
                        der_dec_var_names.append("consumptionElectricityStorage[*,*,*,{}]".format(stage - 1))
                        der_dec_var_names.append("storageLevelElectricityStorage[*,*,*,{}]".format(stage - 1))

                    der_dec_var_names.append("flow1[*,*,*,{}]".format(stage - 1))
                    der_dec_var_names.append("flow2[*,*,*,{}]".format(stage - 1))

                G.add_node(
                    scenario_node_name,
                    cost=cost_var_name,
                    variables=dec_var_names,
                    derived_variables=der_dec_var_names,
                )

                if stage > 1:
                    (sysop_sc, inv_sc) = self.parseScenarioInformation(stage, node, n_stages + 1, n_inv_sc, n_sysop_sc)
                    if stage == n_stages + 1:
                        parent_node = math.ceil(node / n_sysop_sc)
                        tmp_scenario_probability = scenario_probability["sysop_" + str(stage - 1)][sysop_sc[-1] - 1]
                    else:
                        parent_node = math.ceil(node / (n_inv_sc * n_sysop_sc))
                        tmp_scenario_probability = scenario_probability["inv_" + str(stage - 1)][inv_sc[-1] - 1] * scenario_probability["sysop_" + str(stage - 1)][sysop_sc[-1] - 1]

                    parent_scenario_node_name = "sc_{}_{}".format(stage - 1, parent_node)

                    # print(node, sysop_sc[-1], inv_sc[-1], tmp_scenario_probability)

                    G.add_edge(
                        parent_scenario_node_name,
                        scenario_node_name,
                        weight=tmp_scenario_probability,
                    )

                    del (
                        parent_node,
                        parent_scenario_node_name,
                        tmp_scenario_probability,
                        sysop_sc,
                        inv_sc,
                    )

                # print(scenario_node_name, "\n", cost_var_name, "\n", dec_var_names, "\n", der_dec_var_names)
                del (
                    scenario_node_name,
                    cost_var_name,
                    dec_var_names,
                    der_dec_var_names,
                )
            # print("\n")

        # try:
        #     networkx.draw_networkx(G, pos=networkx.spring_layout(G),
        #                            node_size=20, font_size=5, with_labels=True)
        #     plt.savefig(os.path.join(path_result_dir, 'scenarioTree.pdf'), bbox_inches='tight')
        #     print("### EMPRISE scenario tree plot saved as 'scenarioTree.pdf'. ###")
        # except:
        #     print("\n\n!!! EMPRISE scenario tree graph could not be plotted or saved. !!!\n\n")

        return G

    def extractResultData(self, result_data, result_ph, model=None):
        import itertools

        if model is not None:

            scenario_names = [sc for sc in result_ph["scenario solutions"].keys() if sc.startswith("sc_")]

            # - generation thermal capacity investments
            idx = pd.MultiIndex.from_product(
                [[node for node in model.NODE], [gt for gt in model.GEN_THERMAL_TYPE]],
                names=["Node", "generationThermal type"],
            )
            col = model.STAGE
            result_data.result_generation_thermal_capacity = pd.DataFrame(float("nan"), idx, col)
            result_data.result_generation_thermal_capacity_expansion = pd.DataFrame(float("nan"), idx, col)
            for g, st in itertools.product(model.GEN_THERMAL, model.STAGE):
                result_data.result_generation_thermal_capacity.loc[(model.generationThermalNode[g], model.generationThermalType[g]), st] = np.mean(
                    [result_ph["scenario solutions"][sc]["variables"]["generatorThermalCapacity[" + str(g) + "," + str(st) + "]"]["value"] for sc in scenario_names]
                )
                print(([result_ph["scenario solutions"][sc]["variables"]["generatorThermalCapacity[" + str(g) + "," + str(st) + "]"]["value"] for sc in scenario_names]))
                result_data.result_generation_thermal_capacity_expansion.loc[(model.generationThermalNode[g], model.generationThermalType[g]), st] = np.mean(
                    [result_ph["scenario solutions"][sc]["variables"]["generatorThermalNewCapacity[" + str(g) + "," + str(st) + "]"]["value"] for sc in scenario_names]
                )

            # - generation renewable capacity investments
            idx = pd.MultiIndex.from_product(
                [
                    [node for node in model.NODE],
                    [gt for gt in model.GEN_RENEWABLE_TYPE],
                ],
                names=["Node", "generationRenewable type"],
            )
            col = model.STAGE
            result_data.result_generation_renewable_capacity = pd.DataFrame(float("nan"), idx, col)
            result_data.result_generation_renewable_capacity_expansion = pd.DataFrame(float("nan"), idx, col)
            for g, st in itertools.product(model.GEN_RENEWABLE, model.STAGE):
                result_data.result_generation_renewable_capacity.loc[
                    (
                        model.generationRenewableNode[g],
                        model.generationRenewableType[g],
                    ),
                    st,
                ] = np.mean([result_ph["scenario solutions"][sc]["variables"]["generatorRenewableCapacity[" + str(g) + "," + str(st) + "]"]["value"] for sc in scenario_names])
                result_data.result_generation_renewable_capacity_expansion.loc[
                    (
                        model.generationRenewableNode[g],
                        model.generationRenewableType[g],
                    ),
                    st,
                ] = np.mean([result_ph["scenario solutions"][sc]["variables"]["generatorRenewableNewCapacity[" + str(g) + "," + str(st) + "]"]["value"] for sc in scenario_names])

            # - Cost investments and system operation
            idx_inv = ["Investment cost"]
            idx_sys_op = pd.MultiIndex.from_product(
                [["System operation cost"], scenario_names],
                names=["System operation cost", "Scenario"],
            )
            col = model.STAGE
            result_data.result_cost_investment = pd.DataFrame(float("nan"), idx_inv, col)
            result_data.result_cost_system_operation = pd.DataFrame(float("nan"), idx_sys_op, col)
            for st in model.STAGE:
                result_data.result_cost_investment.loc[idx_inv, st] = np.mean([result_ph["scenario solutions"][sc]["variables"]["costInvestment[" + str(st) + "]"]["value"] for sc in scenario_names])
                for sc in scenario_names:
                    result_data.result_cost_system_operation.loc[("System operation cost", sc), st] = result_ph["scenario solutions"][sc]["variables"]["costSystemOperation[" + str(st + 1) + "]"]["value"]
            # print(result_data.result_cost_investment, result_data.result_cost_system_operation)

        else:
            raise Exception("\n!!! EMPRISE model instance is not available !!!\n")

        return result_data

    @staticmethod
    def getCapitalRecoveryFactor(interest_rate, n_periods):
        """Repeating payment factor (annuity)"""
        if interest_rate == 0.0:
            rpf = 1.0
        else:
            rpf = interest_rate * ((1 + interest_rate) ** n_periods) / (((1 + interest_rate) ** n_periods) - 1)
        return rpf

    @staticmethod
    def getMultiPeriodInformation(
        capex: list,
        opex_fix: list,
        wacc: float,
        present_value_interest: float,
        depreciation_period: int,
        investment_period: int,
        number_of_investment_periods: int,
        investment_period_length: int,
        create_plot: bool = False,
    ):
        """Calculates multi-period specific investment cost and decommissioning redemption values

        For multi-period investment planning, this function calculates the total costs (discounted investment and fixed operation)
        starting from an investment period (assuming reinvestments after each depreciation period).
        The function further calculates redemption payments resulting from potential decommissioning decisions in the investment periods following the investment decision.

        Parameters
        ----------
        capex : list (of non-negative floats)
            The capital expenditures for all investment period (in specfic monetary unit, e.g. EUR/MW), len(capex) needs to equal number_of_investment_periods, e.g. [1000.0, 900.0, 750.0, 700.0]
        opex_fix : list (of non-negative floats)
            The fixed operational expenditures for all investment periods (in specfic monetary unit/time period unit, e.g. EUR/MW/yr), len(opex_fix) needs to equal number_of_investment_periods, e.g. [50.0, 50.0, 50.0, 50.0]
        wacc : float (in the interval [0,1])
            Weighted average cost of capital (in 1) to calculate equivalent annual cost, e.g. 0.06
        present_value_interest : float (in the interval [0,1])
            Present value interest rate (in 1) to calculate the discount factors representing the time value of money including the perpetuity, e.g. 0.02
        depreciation_period : float (non-negative)
            Depreciation period (in years) to calculate equivalent annual cost, e.g. 12
        investment_period : int (non-negative and smaller or equal to number_of_investment_periods)
            Investment period for which the first investment decision is assumed, e.g. 2
        number_of_investment_periods : int (non-negative)
            Total number of investment periods including perpetuity investment period, e.g. 4
        investment_period_length : int (non-negative)
            Length of the equal-length investment periods (in time units, e.g. years), e.g. 10
        create_plot : bool
            Plot total cost results over planning horizon

        Output
        ------
        total_cost : float
            Total (discounted) investment and fixed operation cost for investment decision in investment_period until the end of time (perpetuity in last considered period)
        decommissioning_redemption : list (of floats)
            Redemption payments for potential future decommissioning decisions starting after the investment_period to compensate the total cost paid for the entirety of the planning horizon
        operational_stage_contribution_matrix : two-dimensional numpy array (of floats)
            Share of technology development (investment) period (col) contribution for every considered operation period (row), including discount factor information
        # CURRENTLY NOT USED: operational_stage_idx : two-dimensional numpy array (of floats)
            Share of technology development (investment) period (col) contribution for every considered operation period (row), excluding discount factor information. NOTE: Perpetuity currently counts as a single year

        Example
        -------
        getMultiPeriodInformation([1000.0, 900.0, 750.0, 700.0], [50.0, 50.0, 50.0, 50.0], [0.06, 0.06, 0.06, 0.06], 0.02, [15, 13, 12, 10], 2, 4, 10, False)
        -> total_cost = 5951.8285
        -> decommissioning_redemption = [4591.0051, 3708.4673]
        -> operational_stage_contribution_matrix = [[0.         0.         0.         0.        ]
                                                    [0.         1.         0.         0.        ]
                                                    [0.         0.35372757 0.64627243 0.        ]
                                                    [0.         0.         0.0960792  0.9039208 ]]
        """

        # print(capex, opex_fix, wacc, present_value_interest, depreciation_period, investment_period, number_of_investment_periods, investment_period_length, create_plot)
        # print(
        #     type(capex), type(opex_fix), type(wacc), type(present_value_interest), type(depreciation_period), type(investment_period), type(number_of_investment_periods), type(investment_period_length), type(create_plot)
        # )

        import numpy as np
        import math

        if capex is None:
            capex = [1.0 for i in range(number_of_investment_periods)]
        if opex_fix is None:
            opex_fix = [1.0 for i in range(number_of_investment_periods)]
        if wacc is None:
            wacc = [0.05 for i in range(number_of_investment_periods)]

        if create_plot:
            import matplotlib.pyplot as plt

        equivalent_annual_cost_factor = [wacc[x] / (1 - (1 + wacc[x]) ** -depreciation_period[x]) for x in range(len(capex))]
        perpetuity_factor = 1.0 / present_value_interest
        discount_factors = (1 - present_value_interest) ** np.arange(2 * (number_of_investment_periods) * investment_period_length)

        # Determine (re)investment periods
        start_index = [
            x for x in range(investment_period - 1, len(depreciation_period)) if 1 + sum(depreciation_period[investment_period - 1 : x]) <= investment_period_length * (number_of_investment_periods - investment_period)
        ]
        investment_start = [((investment_period - 1) * investment_period_length) + 1 + sum(depreciation_period[investment_period - 1 : x]) for x in start_index]
        if not start_index:  # list is empty (implicit booleanness)
            investment_start_perpetuity = ((investment_period - 1) * investment_period_length) + 1
        else:
            investment_start_perpetuity = ((investment_period - 1) * investment_period_length) + 1 + sum(depreciation_period[investment_period - 1 : start_index[-1] + 1])
        investment_start_all = investment_start + [investment_start_perpetuity]

        # Retrieve investment and fixed operation cost
        investment_cost_matrix = np.zeros((len(investment_start) + 1, len(discount_factors)))
        fixed_operation_cost_matrix = np.zeros((len(investment_start) + 1, len(discount_factors)))
        operational_stage_matrix = np.zeros((len(investment_start) + 1, len(discount_factors)))
        operational_stage_idx_matrix = np.zeros((len(investment_start) + 1, len(discount_factors)))
        operational_stage_discount_factor_matrix = np.zeros((len(investment_start) + 1, len(discount_factors)))
        for inv in range(len(investment_start)):
            tmp_investment_period = math.floor(investment_start[inv] / investment_period_length)
            investment_cost_matrix[inv, (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)] = (
                capex[tmp_investment_period]
                * equivalent_annual_cost_factor[tmp_investment_period]
                * discount_factors[(investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)]
            )
            fixed_operation_cost_matrix[inv, (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)] = (
                opex_fix[tmp_investment_period] * discount_factors[(investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)]
            )
            operational_stage_matrix[inv, (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)] = tmp_investment_period + 1
            operational_stage_discount_factor_matrix[inv, (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)] = discount_factors[
                (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)
            ]
            operational_stage_idx_matrix[inv, (investment_start[inv] - 1) : (investment_start[inv] + depreciation_period[tmp_investment_period] - 1)] = 1.0
            del tmp_investment_period
            if create_plot:
                plt.bar(range(len(discount_factors)), investment_cost_matrix[inv, :])
                plt.bar(range(len(discount_factors)), fixed_operation_cost_matrix[inv, :], bottom=investment_cost_matrix[inv, :])

        tmp_investment_period = math.floor(investment_start_perpetuity / investment_period_length)
        if tmp_investment_period > len(capex) - 1:
            tmp_investment_period = len(capex) - 1
        investment_cost_matrix[-1, investment_start_perpetuity - 1] = (
            capex[tmp_investment_period] * equivalent_annual_cost_factor[tmp_investment_period] * perpetuity_factor * discount_factors[investment_start_perpetuity - 1]
        )
        fixed_operation_cost_matrix[-1, investment_start_perpetuity - 1] = opex_fix[tmp_investment_period] * perpetuity_factor * discount_factors[investment_start_perpetuity - 1]
        operational_stage_matrix[-1, investment_start_perpetuity - 1] = tmp_investment_period + 1
        operational_stage_discount_factor_matrix[-1, investment_start_perpetuity - 1] = perpetuity_factor * discount_factors[investment_start_perpetuity - 1]
        operational_stage_idx_matrix[-1, investment_start_perpetuity - 1] = 1.0  # TODO: Needs to be corrected, current value does not make much sense
        del tmp_investment_period

        # Create plot if chosen by the user
        if create_plot:
            plt.bar(range(len(discount_factors)), investment_cost_matrix[-1, :])
            plt.bar(range(len(discount_factors)), fixed_operation_cost_matrix[-1, :], bottom=investment_cost_matrix[-1, :])
            plt.xlabel("Planning horizon")
            plt.ylabel("Cost in monetary unit/time unit")
            plt.title("Annual investment and fixed operation cost")
            plt.show()

        # Total cost as the sum of all (discounted) investment and fixed operation cost
        total_cost = sum(sum(investment_cost_matrix)) + sum(sum(fixed_operation_cost_matrix))

        # Compile decommissioning redemption values for each relevant decision period
        decommissioning_redemption_investment = []
        decommissioning_redemption_operation = []
        for ip in range(investment_period, number_of_investment_periods):
            idx1 = next(x for x, val in enumerate(investment_start_all) if val >= ip * investment_period_length)
            decommissioning_redemption_investment.append(sum(sum(investment_cost_matrix[idx1:, :])))  # Redemption payments from omitted reinvestments may only come from
            tmp_redemption_operation = [sum(sum(fixed_operation_cost_matrix[:, idx2 * investment_period_length : (idx2 + 1) * investment_period_length - 1])) for idx2 in range(ip, number_of_investment_periods)]
            decommissioning_redemption_operation.append(sum(tmp_redemption_operation))
            del tmp_redemption_operation, idx1

        # Compile weigths for operational stages
        operational_stage_contribution = np.zeros((number_of_investment_periods, number_of_investment_periods))
        operational_stage_idx = np.zeros((number_of_investment_periods, number_of_investment_periods))
        for ip in range(0, number_of_investment_periods):
            for idx3 in range(0, number_of_investment_periods):
                tmp_operational_stage_matrix = operational_stage_matrix[:, ip * investment_period_length : (ip + 1) * investment_period_length - 1].astype(int)
                tmp_operational_stage_discount_factor_matrix = operational_stage_discount_factor_matrix[:, ip * investment_period_length : (ip + 1) * investment_period_length - 1]
                tmp_operational_stage_idx_matrix = operational_stage_idx_matrix[:, ip * investment_period_length : (ip + 1) * investment_period_length - 1]
                operational_stage_contribution[ip, idx3] = tmp_operational_stage_discount_factor_matrix[tmp_operational_stage_matrix == idx3 + 1].sum()
                operational_stage_idx[ip, idx3] = tmp_operational_stage_idx_matrix[tmp_operational_stage_matrix == idx3 + 1].sum()
            if operational_stage_contribution[ip, :].sum() > 0.0:
                operational_stage_contribution[ip, :] = operational_stage_contribution[ip, :] / operational_stage_contribution[ip, :].sum()
                operational_stage_idx[ip, :] = operational_stage_idx[ip, :] / operational_stage_idx[ip, :].sum()

        # print(decommissioning_redemption_investment, decommissioning_redemption_operation)

        # Total decommissioning redemption payments
        decommissioning_redemption = [sum(x) for x in zip(decommissioning_redemption_investment, decommissioning_redemption_operation)]

        return (total_cost, decommissioning_redemption, operational_stage_contribution)  # operational_stage_idx currently not used

    @staticmethod
    def getDiscountFactorSum(interest_rate, stage, years_per_stage, allocation_method=""):
        """Calculates the stage-specific discount factor as a sum over all years belonging to the decision stage.
        The allocation method determines how the summation window is configured, i.e. does the given decision stage refer to the starting year or the centre of the years
        """
        import numpy as np

        relevant_years = np.arange((stage - 1) * years_per_stage, stage * years_per_stage)
        if allocation_method == "center":
            relevant_years = relevant_years - int(years_per_stage / 2)
        discount_factor = 1 / (1 + interest_rate) ** relevant_years
        # discount_factor[discount_factor > 1] = 1  # relevant for years before current year (years_per_stage/2), only for center method
        discount_factor_sum = sum(discount_factor)
        discount_factor_perpetuity = discount_factor[-1] / interest_rate
        return (discount_factor_sum, discount_factor_perpetuity)

    @staticmethod
    def parseScenarioInformation(stage, node, n_stages, n_inv_sc, n_sysop_sc):
        """Get investment and system operation scenario information (given an balanced/symmetric branching structure)"""
        from math import ceil

        if stage > 1:
            if stage == n_stages:
                number_of_branches_per_node = n_sysop_sc
                sysop_sc = node - (ceil(node / number_of_branches_per_node) - 1) * number_of_branches_per_node
                inv_sc = 1  # arbitrary since not relevant for last stage of scenario tree
            else:
                number_of_branches_per_node = n_inv_sc * n_sysop_sc
                inv_sc_help = node - (ceil(node / number_of_branches_per_node) - 1) * number_of_branches_per_node
                sysop_sc = inv_sc_help - (ceil(inv_sc_help / n_sysop_sc) - 1) * n_sysop_sc
                inv_sc = ceil(inv_sc_help / n_sysop_sc)

            (sysop_sc_prev, inv_sc_prev) = EmpriseModel.parseScenarioInformation(
                stage - 1,
                ceil(node / number_of_branches_per_node),
                n_stages,
                n_inv_sc,
                n_sysop_sc,
            )
            inv_sc = inv_sc_prev + [inv_sc]
            sysop_sc = sysop_sc_prev + [sysop_sc]
        else:
            sysop_sc = [1]
            inv_sc = [1]
        return (sysop_sc, inv_sc)

    @staticmethod
    def truncate(x, precision=4):
        return x  # int(x * 10 ** precision) / 10 ** precision
