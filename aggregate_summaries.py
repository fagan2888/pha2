from glob import glob
import os

#filespec = 'outputs/summary_543259_Scenario_*.tsv'
filespec = 'outputs/summary_543361_Scenario_*.tsv'
outfile = 'outputs/summary_543361_aggregated.tsv'

infiles = glob(filespec)
wrote_header = False
print "creating {}...".format(outfile)
with open(outfile, 'w') as fout:    
    for infile in infiles:
        print "reading {}...".format(infile)
        with open(infile) as fin:
            rows = [row.split('\t') for row in fin.read().splitlines()]
            cols = zip(*rows)
            if not wrote_header:
                fout.write('\t'+ '\t'.join(cols[0]) + '\n')
                fout.write('\t'+ '\t'.join(cols[1]) + '\n')
                wrote_header = True
            fout.write(infile + '\t' + '\t'.join(cols[2])+'\n')

#filespec = 'outputs/build_543259_iter0_Scenario_*.tsv'
filespec = 'outputs/build_543361_iter0_Scenario_*.tsv'
outfile = 'outputs/build_543361_aggregated.tsv'

infiles = glob(filespec)
wrote_header = False
print "creating {}...".format(outfile)
with open(outfile, 'w') as fout:    
    for infile in infiles:
        print "reading {}...".format(infile)
        with open(infile) as fin:
            rows = [row.split('\t') for row in fin.read().splitlines()]
            cols = zip(*rows)
            if not wrote_header:
                fout.write('\t'+ '\t'.join(cols[0]) + '\n')
                wrote_header = True
            fout.write(infile + '\t' + '\t'.join(cols[1])+'\n')
