from glob import glob
from collections import defaultdict
import numpy as np

builds = [
    # ("b422385_iter0", "cross-scenario tests"),
    # ("b416974_xhat", "optimal plan"),
    # ("422384_xhat", "mean fuel prices"),
    ("_433934_", "PSIP preferred plan - mostly dist PV - first solve"),
    ("434551_b433934_iter0", "PSIP preferred plan - mostly dist PV - all scenarios"),
    # ("_434414_", "PSIP preferred plan - more central PV - first solve"),
    # ("434490_b434414_iter0", "PSIP preferred plan - more central PV - all scenarios"),
]

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
for group, build_list in build_scenarios.iteritems():
    for scenario, file_list in build_list.iteritems():
        # read files, get average and summary statistics
        print "reading values for {}: {}".format(group, scenario)
        vals = defaultdict(list)
        for file in file_list:
            with open(file, "r") as f:
                data = [r[:-1].split("\t") for r in f]  # split at tabs; omit newline
                for r in data:
                    key = (r[0], r[1])
                    vals[key].append(float(r[2]))   # index by key and period, tabulate values
                    if key not in keys:
                        keys.append(key)
        results_mean[group, scenario] = {k: np.mean(v) for k, v in vals.iteritems()}
        results_05[group, scenario] = {k: np.percentile(v, 5) for k, v in vals.iteritems()}
        results_95[group, scenario] = {k: np.percentile(v, 95) for k, v in vals.iteritems()}

with open("outputs/summary_all_scenarios.tsv", "w") as f:
    f.write("group\tscenario\t" + "\t".join(
        ["_".join(key) for key in keys] 
        + ["_".join(key)+"_05" for key in keys]
        + ["_".join(key)+"_95" for key in keys]
    ) + "\n")
    for tag, group in builds:
        for scenario in build_scenarios[group].keys():
            f.write("\t".join(
                [group, scenario]
                + [str(results_mean[group, scenario][key]) for key in keys]
                + [str(results_05[group, scenario][key]) for key in keys]
                + [str(results_95[group, scenario][key]) for key in keys]
            ) + "\n")
