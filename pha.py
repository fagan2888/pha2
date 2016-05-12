""" adjustments and utility methods to support pha algorithm """

import os, types
from pyomo.environ import *
import switch_mod.utilities as utilities

build_vars = [
    "BuildProj", "BuildUnits", 
    "ConvertToLNG",
    "BuildBattery", 
    "BuildPumpedHydroMW", "BuildAnyPumpedHydro",
    "RFMSupplyTierActivate",
    "BuildElectrolyzerMW", "BuildLiquifierKgPerHour", "BuildLiquidHydrogenTankKg",
    "BuildFuelCellMW"
]

n_digits = 4    # zero-padding in existing file names and scenario names

def define_arguments(argparser):
    # define data location and size
    # NOTE: these are not used when the model is loaded by runph;
    # they are only used when ReferenceModel is run as a script
    # to create runph input files.
    argparser.add_argument("--pha-subdir", default='pha',
        help="Subdirectory of inputs directory to store PHA data files in.")
    argparser.add_argument("--pha-scenario-count", type=int, default=1,
        help="Number of scenarios to use for PHA solution.")

def define_components(m):
    # add methods to support PHA modeling
    m.save_pha_dat_files = types.MethodType(save_pha_dat_files, m)
    m.save_pha_rho_file = types.MethodType(save_pha_rho_file, m)

def define_dynamic_components(m):
    # add dummy expressions to keep runph happy
    # note: it doesn't seem to be necessary to 
    # apportion the full objective function between these
    # also note: we do this in define_dynamic_components because SystemCost is not
    # defined until that point in financials.py
    m.BuildCost = Expression(rule=lambda m: 0.0)
    m.OperateCost = Expression(rule=lambda m: m.SystemCost)


def dat_file_dir(m):
    return os.path.join(m.options.inputs_dir, m.options.pha_subdir)

def save_pha_dat_files(m):
    n_scenarios = m.options.pha_scenario_count

    dat_file = os.path.join(dat_file_dir(m), "RootNode.dat")
    print "saving {}...".format(dat_file)
    utilities.save_inputs_as_dat(
        model=m, instance=m, save_path=dat_file,
        exclude=["rfm_supply_tier_cost", "rfm_supply_tier_limit", "rfm_supply_tier_fixed_cost"])

    scenarios = [str(i).zfill(n_digits) for i in range(n_scenarios)]
    
    dat_file = os.path.join(dat_file_dir(m), "ScenarioStructure.dat")
    print "saving {}...".format(dat_file)
    with open(dat_file, "w") as f:
        # use show only the changed data in the dat files for each scenario
        f.write("param ScenarioBasedData := False ;\n\n")
        
        f.write("set Stages := Build Operate ;\n\n")

        f.write("set Nodes := RootNode \n")
        for s in scenarios:
            f.write("    fuel_supply_curves_{}\n".format(s))
        f.write(";\n\n")

        f.write("param NodeStage := RootNode Build\n")
        for s in scenarios:
            f.write("    fuel_supply_curves_{} Operate\n".format(s))
        f.write(";\n\n")
        
        f.write("set Children[RootNode] := \n")
        for s in scenarios:
            f.write("    fuel_supply_curves_{}\n".format(s))
        f.write(";\n\n")
    
        f.write("param ConditionalProbability := RootNode 1.0\n")
        probs = [1.0/n_scenarios] * (n_scenarios - 1) # evenly spread among all scenarios
        probs.append(1.0 - sum(probs))  # lump the remainder into the last scenario
        for (s, p) in zip(scenarios, probs):
            f.write("    fuel_supply_curves_{s} {p}\n".format(s=s, p=p))
        f.write(";\n\n")

        f.write("set Scenarios :=  \n")
        for s in scenarios:
            f.write("    Scenario_{}\n".format(s))
        f.write(";\n\n")

        f.write("param ScenarioLeafNode := \n")
        for s in scenarios:
            f.write("    Scenario_{s} fuel_supply_curves_{s}\n".format(s=s, p=p))
        f.write(";\n\n")

        def write_var_name(f, cname):
            if hasattr(m, cname):
                dimen = getattr(m, cname).index_set().dimen
                indexing = "" if dimen == 0 else (",".join(["*"]*dimen))
                f.write("    {cn}[{dim}]\n".format(cn=cname, dim=indexing))

        # all build variables (and fuel market expansion) go in the Build stage
        f.write("set StageVariables[Build] := \n")
        for cn in build_vars:
            write_var_name(f, cn)
        f.write(";\n\n")
        
        # all other variables go in the Operate stage
        operate_vars = [
            c.cname() for c in m.component_objects() 
                if isinstance(c, pyomo.core.base.var.Var) and c.cname() not in build_vars
        ]
        f.write("set StageVariables[Operate] := \n")
        for cn in operate_vars:
            write_var_name(f, cn)
        f.write(";\n\n")

        f.write("param StageCostVariable := \n")
        f.write("    Build BuildCost\n")
        f.write("    Operate OperateCost\n")
        f.write(";\n\n")
        # note: this uses dummy variables for now; if real values are needed,
        # it may be possible to construct them by extracting all objective terms that 
        # involve the Build variables.

def save_pha_rho_file(m):
    print "calculating objective function coefficients for rho setters..."
    # initialize variables, if not already set
    for var in m.component_map(Var):
        for v in getattr(m, var).values():
            if v.value is None:
                # note: we're just using this to find d_objective / d_var,
                # so it doesn't need to be realistic or even within the allowed bounds
                # if the model is linear; 
                v.value = 0.0

    costs = []
    baseval = value(m.Minimize_System_Cost)
    # surprisingly slow, but it gets the job done
    for var in build_vars:
        print var
        for v in getattr(m, var).values():
            # perturb the value of each variable to find its coefficient in the objective function
            v.value += 1; c = value(m.Minimize_System_Cost) - baseval; v.value -= 1
            costs.append((v.cname(), c))
    rho_file = os.path.join(m.options.inputs_dir, "rhos.tsv")
    print "writing {}...".format(rho_file)
    with open(rho_file, "w") as f:
        f.writelines("\t".join(map(str, r))+"\n" for r in costs)
