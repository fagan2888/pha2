from glob import glob
import os

infiles = 'outputs/summary_543259_Scenario_*.tsv'
outfile = 'outputs/summary_543259_aggregated.tsv'

match_files = glob(filespec)
vals = defaultdict(list)
wrote_header = False
print "creating {}...".format(outfile)
with open(outfile, 'w') as out:    
    for infile in infiles:
        print "reading {}...".format(infile)
        with open(infile) as in:
            rows = [row.split('\t') for row in.read().splitlines()]
            cols = zip(*rows)
            if not wrote_header:
                out.write('\t'+ '\t'.join(cols[0]) + '\n')
                out.write('\t'+ '\t'.join(cols[1]) + '\n')
                wrote_header = True
            out.write(infile + '\t' + '\t'.join(cols[2])+'\n')


infiles = 'outputs/build_543259_iter0_Scenario_*.tsv'
outfile = 'outputs/build_543259_aggregated.tsv'

match_files = glob(filespec)
vals = defaultdict(list)
wrote_header = False
print "creating {}...".format(outfile)
with open(outfile, 'w') as out:    
    for infile in infiles:
        print "reading {}...".format(infile)
        with open(infile) as in:
            rows = [row.split('\t') for row in.read().splitlines()]
            cols = zip(*rows)
            if not wrote_header:
                out.write('\t'+ '\t'.join(cols[0]) + '\n')
                wrote_header = True
            out.write(infile + '\t' + '\t'.join(cols[1])+'\n')
