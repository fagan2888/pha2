from glob import glob
from collections import defaultdict
import numpy as np
import os

inputs_dir = 'inputs'
pha_subdir = 'pha_125_logistic'
pha_dir = os.path.join(inputs_dir, pha_subdir)

builds = [
    # (file tag, group name)
    ("b540396_iter0", "PSIP preferred plan"),
    # ("b540398_iter0", "mean fuel prices"),
    ("b540397_iter0", "cross-scenario tests"),
    # ("b540397test_iter0", "cross-scenario tests"),
]

# read the weights for the scenarios
scenario_weight = dict()
with open(os.path.join(pha_dir, 'scenario_weights.tsv')) as f:
    f.next()    # skip headers
    data = [r[:-1].split("\t") for r in f]  # split at tabs; omit newline
    scenario_weight = {int(s): float(w) for s, w in data}

build_scenarios = {}
for tag, group in builds:
    match_files = glob("outputs/summary*{}*.tsv".format(tag))
    build_list = defaultdict(list)
    for f in match_files:
        # use the part of the filename between the tag and the final "_Scenario_nnnn"
        # as the unique name of this scenario. For the xhat builds, this will just be 
        # the same as the tag. For the iter0 builds (which end with "Scenario_nnnn_Scenario_nnnn"),
        # this will include the name of the original scenario.
        scenario = f[f.index(tag):f.rfind('_Scenario_')]
        build_list[scenario].append(f)
    build_scenarios[group] = build_list

# now build_scenarios contains a list of case studies, each of which contains 
# one or more scenarios, each of which contains names of many files showing the 
# results of that scenario
keys = []
results_mean = {}
results_05 = {}
results_95 = {}

# read data from summary files
for group, build_list in build_scenarios.iteritems():
    for scenario, file_list in build_list.iteritems():
        # read files, get average and summary statistics
        print "reading values for {}: {}".format(group, scenario)
        vals = defaultdict(list)
        eval_scen_ids = []   # list of scenarios used for evaluation
        for file in file_list:
            # identify the price scenario which was used for this evaluation 
            # omit '.tsv' from end of the file name, then take the part after the last '_'
            eval_scen_ids.append(int(file[:-4].split('_')[-1]))
            with open(file, "r") as f:
                data = [r[:-1].split("\t") for r in f]  # split at tabs; omit newline
                for r in data:
                    key = (r[0], r[1])  # parameter, year
                    vals[key].append(float(r[2]))  # index by key and period, tabulate values
                    if key not in keys:
                        keys.append(key)    # this retains the original reporting order
        weight = np.array([scenario_weight[id] for id in eval_scen_ids])
        weight = weight / weight.sum()  # normalize; probably not needed, but just in case
        results_mean[group, scenario] = dict()
        results_05[group, scenario] = dict()
        results_95[group, scenario] = dict()
        for k, v in vals.iteritems():
            val = np.array(v)   # vector of values for this key
            results_mean[group, scenario][k] = float(np.sum(val * weight))
            sort_idx = np.argsort(val)
            val_sort = val[sort_idx]
            val_rank = weight[sort_idx].cumsum() - 0.5 * weight[sort_idx]
            results_05[group, scenario][k] = np.interp(x=0.05, xp=val_rank, fp=val_sort)
            results_95[group, scenario][k] = np.interp(x=0.95, xp=val_rank, fp=val_sort)


with open("outputs/summary_all_scenarios_weighted.tsv", "w") as f:
    f.write("group\tscenario\t" + "\t".join(
        ["_".join(key) for key in keys] 
        + ["_".join(key)+"_05" for key in keys]
        + ["_".join(key)+"_95" for key in keys]
    ) + "\n")
    for tag, group in builds:
        # write a row for each scenario in each group, sorted by scenario name (number)
        for scenario in sorted(build_scenarios[group].keys()):
            try:
                f.write("\t".join(
                    [group, scenario]
                    + [str(results_mean[group, scenario][key]) for key in keys]
                    + [str(results_05[group, scenario][key]) for key in keys]
                    + [str(results_95[group, scenario][key]) for key in keys]
                ) + "\n")
            except KeyError:
                print "WARNING: KeyError for results_xx[{g}, {s}][{k}].".format(g=group, s=scenario, k=key)
                f.write('\t'.join([group, scenario, 'MISSING DATA']) + '\n')
