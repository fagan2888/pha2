from glob import glob
from collections import defaultdict
import numpy as np
import os

outfile = "outputs/summary_selected_scenarios_05_95.tsv"
inputs_dir = 'inputs'
pha_subdir = 'pha_125_logistic'
pha_dir = os.path.join(inputs_dir, pha_subdir)

builds = [
    # (file tag, group name)
    # ('20160517_084856_psip_preferred', 'PSIP'),
    # ('20160517_084916_high_lng', 'High LNG'),
    # ('20160517_084934_high_lsfo', 'High LSFO'),
    # ('20160517_084856_high_re', 'High RE'),
    # ("542667_b540876_iter0", "PSIP Theme 2 Preferred Plan"),
    # ("542692_b540877_iter0", "cross-scenario tests"),
    # ("542693_b540877_iter0", "cross-scenario tests"),
    # ("542694_b540877_iter0", "cross-scenario tests"),
    # ("542695_b540877_iter0", "cross-scenario tests"),
    # ("b540398_iter0", "mean fuel prices"),
    ("543262_b540876_iter0", "PSIP Theme 2 Preferred Plan"),
    ("b543259_iter0", "selected optimal plans"),
]

# read the weights for the scenarios
scenario_weight = dict()
with open(os.path.join(pha_dir, 'scenario_weights.tsv')) as f:
    f.next()    # skip headers
    data = [r[:-1].split("\t") for r in f]  # split at tabs; omit newline
    scenario_weight = {int(s): float(w) for s, w in data}

build_scenarios = defaultdict(dict)
for tag, group in builds:
    match_files = glob("outputs/summary*{}*.tsv".format(tag))
    build_list = defaultdict(list)
    for f in match_files:
        # use the part of the filename between the tag and the final "_Scenario_nnnn"
        # as the unique name of this scenario. For the xhat builds, this will just be 
        # the same as the tag. For the iter0 builds (which end with "Scenario_nnnn_Scenario_nnnn"),
        # this will include the name of the original scenario.
        scenario = f[f.index(tag):f.rfind('_Scenario_')]
        build_list[scenario].append(f)  # add filename to end of list for this scenario
    build_scenarios[group].update(build_list)   # add all scenarios to build_scenarios list

# now build_scenarios contains a list of case studies, each of which contains 
# one or more scenarios, each of which contains names of many files showing the 
# results of that scenario
keys = []
results_mean = {}
results_low = {}
results_high = {}

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
        results_low[group, scenario] = dict()
        results_high[group, scenario] = dict()
        for k, v in vals.iteritems():
            val = np.array(v)   # vector of values for this key
            results_mean[group, scenario][k] = float(np.sum(val * weight))
            sort_idx = np.argsort(val)
            val_sort = val[sort_idx]
            val_rank = weight[sort_idx].cumsum() - 0.5 * weight[sort_idx]
            results_low[group, scenario][k] = np.interp(x=0.05, xp=val_rank, fp=val_sort)
            results_high[group, scenario][k] = np.interp(x=0.95, xp=val_rank, fp=val_sort)

print "writing {}...".format(outfile)
with open(outfile, "w") as f:
    row = ['group', 'scenario']
    for key in keys:
        k = '_'.join(key)
        row.extend([k, k+'_up', k+'_down'])
    f.write('\t'.join(row)+ '\n')
    for tag, group in builds:
        # write a row for each scenario in each group, sorted by scenario name (number)
        for scenario in sorted(build_scenarios[group].keys()):
            row = [group, scenario]
            for key in keys:
                try:
                    mean = results_mean[group, scenario][key]
                    low = results_low[group, scenario][key]
                    high = results_high[group, scenario][key]
                    row.extend([mean, high-mean, mean-low])
                except KeyError:
                    print "WARNING: KeyError for results_xx[{}, {}][{}].".format(group, scenario, key)
                    row.extend(['MISSING DATA'] * 3)
            f.write("\t".join(map(str, row)) + '\n')
            # + [str(results_mean[group, scenario][key]) for key in keys]
            # + [str(results_low[group, scenario][key]) for key in keys]
            # + [str(results_high[group, scenario][key]) for key in keys]

