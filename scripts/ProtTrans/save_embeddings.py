#!/usr/bin/env python3.7

# A script that collects ProtTrans embeddings to an
# NPZ file.

import sys
import torch
from optparse import OptionParser
import random
from Bio import SeqIO
import numpy

parser = OptionParser()

parser.add_option('--fasta', '-f', dest='fasta',
    help='path to the data set FASTA file')

parser.add_option('--pt-dir', dest='pt_dir',
    help='directory of input ProtTrans PT (.pt) file(s)')

parser.add_option('--unlabelled', dest='unlabelled', default=False,
    action='store_true', help='option to determine, whether a data set is '+\
    'labelled or not')

parser.add_option('--representations', dest='representations',
    help='chosen representations to save. '+\
        'Options: per_res, mean, std, quantile, hist.')

parser.add_option('--keyword', dest='keyword',
    help='keyword that determines the subset: '+\
        '(i.e. "train", "validate", "test")')

parser.add_option('--tsv', '-t', dest='tsv', default=None,
    help='path to the output TSV file')

(options, args) = parser.parse_args()

options.representations = options.representations.split(' ')
for representation in options.representations:
    if(representation not in ['per_res', 'mean', 'std', 'quantile', 'hist']):
        print("%s: representation option %s is unavailable. Skipping" %\
             (sys.argv[0], representation), file=sys.stderr)
        options.representations.remove(representation)

if(options.keyword == None):
    print(sys.argv[0]+': keyword required.', file=sys.stderr)
    exit

x = [] # keeps embeddings
y = [] # keeps temperatures
z = [] # keeps information about the sequence

for record in SeqIO.parse(options.fasta, 'fasta'):
    if(options.unlabelled):
        seqs_temp = 999
    else:
        seqs_temp = int(record.name.split('|')[2])
    y.append(seqs_temp)
    z.append(record.name+'|'+str(len(record.seq)))

    fn = '%s/%s.pt' % (options.pt_dir, record.name)
    embs = torch.load(fn)

    embedding = embs[options.representations[0]+'_representations']

    embedding = torch.flatten(embedding)

    for i, representation in enumerate(options.representations):
        if(i):
            embedding = torch.cat((embedding,
                torch.flatten(embs[representation+\
                '_representations'])), 0)

    x.append(embedding)

random.seed(27)

tmp = list(zip(x, y, z))
random.shuffle(tmp)
x, y, z = zip(*tmp)

# Saving to a TSV file
if(options.tsv):
    f = open(options.tsv, 'w')
    for i in range(len(z)):
        emb_str = ''

        for j in range(x[i].size(dim=0)):
            emb_str += '%.3f\t' % x[i][j].item()

        if(options.unlabelled):
            # 0th position of z: protein's id, 1st position: length
            f.write('-\t%s\t%s\t999\t%s\n' %
                (z[i].split('|')[0], z[i].split('|')[1], emb_str))
        else:
            seq_header = '|'.join(z[i].split('|')[0:3])

            f.write('%s\t%s\t%s\t%s\t%s\n' %
                (z[i].split('|')[0], z[i].split('|')[1],
                z[i].split('|')[3], z[i].split('|')[2], emb_str))

    f.close()
