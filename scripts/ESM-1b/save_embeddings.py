#!/usr/bin/env python3.7

# A script that collects ESM-1b embeddings to an
# NPZ file

import sys
import torch
from optparse import OptionParser
import random
from Bio import SeqIO
import numpy

EMB_LAYER = 33

parser = OptionParser()

parser.add_option('--fasta', '-f', dest='fasta',
    help='path to the data set FASTA file')

parser.add_option('--pt-dir', dest='pt_dir',
    help='directory of input ESM-1b PT (.pt) file(s)')

parser.add_option('--representations', dest='representations',
    help='chosen representations to save. '+\
        'Options: mean, std, quantile, hist.')

parser.add_option('--keyword', dest='keyword',
    help='keyword that determines the subset: '+\
        '(i.e. "train", "validate", "test")')

parser.add_option('--npz', '-n', dest='npz',
    help='path to the output NPZ file')

parser.add_option('--tsv', '-t', dest='tsv', default=None,
    help='path to the output TSV file')

(options, args) = parser.parse_args()

options.representations = options.representations.split(' ')
for representation in options.representations:
    if(representation not in ['mean', 'std', 'quantile', 'hist']):
        print("%s: representation option %s is unavailable. Skipping." %\
            (sys.argv[0], representation), file=sys.stderr)
        options.representations.remove(representation)

if(options.keyword == None):
    print(sys.argv[0]+': keyword required.', file=sys.stderr)
    exit

x = [] # keeps embeddings
y = [] # keeps temperatures
z = [] # keeps information about the sequence

all_fasta_records = 0
fasta_records_with_embeddings = 0

for record in SeqIO.parse(options.fasta, 'fasta'):
    all_fasta_records += 1
    if(len(record.seq) <= 1022):
        fasta_records_with_embeddings += 1

        seqs_temp = int(record.name.split('|')[2])
        y.append(seqs_temp)
        z.append(record.name+'|'+str(len(record.seq)))

        fn = '%s/%s.pt' % (options.pt_dir, record.name)
        embs = torch.load(fn)

        embedding = embs[options.representations[0]+'_representations'][EMB_LAYER]

        embedding = torch.flatten(embedding)

        for i, representation in enumerate(options.representations):
            if(i):
                embedding = torch.cat((embedding,
                    torch.flatten(embs[representation+\
                    '_representations'][EMB_LAYER])), 0)

        x.append(embedding)

x = torch.stack(x)
y = torch.from_numpy(numpy.asarray(y).astype('int32'))

random.seed(27)

tmp = list(zip(x, y, z))
random.shuffle(tmp)
x, y, z = zip(*tmp)

# Saving to a TSV file
if(options.tsv):
    f = open(options.tsv, 'w')
    for i in range(len(z)):
        emb_str = ''
        seq_header = '|'.join(z[i].split('|')[0:3])
        for j in range(x[i].size(dim=0)):
            emb_str += '%.3f\t' % x[i][j].item()

        f.write('%s\t%s\t%s\t%s\t%s\n' %
            (z[i].split('|')[0], z[i].split('|')[1],
            z[i].split('|')[3], z[i].split('|')[2], emb_str))

    f.close()

print('All FASTA records:\t%d\nFASTA records with embeddings:\t%d' %
        (all_fasta_records, fasta_records_with_embeddings))

# Saving embeddings to an NPZ file
if(options.npz):
    numpy.savez(options.npz, **{name:value for name, value
        in zip(['x_'+options.keyword, 'y_'+options.keyword, 'z_'+options.keyword],
        [x, y, z])})
