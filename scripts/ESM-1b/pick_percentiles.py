#!/usr/bin/env python3.7

# This script picks the required percentiles
# from per token representations to save to 
# the separate TSV file

import sys
from optparse import OptionParser
from Bio import SeqIO
import torch
import random
import numpy

EMB_LAYER = 33

parser = OptionParser()

parser.add_option('--fasta', '-f', dest='fasta',
	help='path to the data set FASTA file')

parser.add_option('--pt-dir', dest='pt_dir',
	help='directory of input PT (.pt) file(s) that have '+\
		'per residue or per token embeddings')

parser.add_option('--keyword', dest='keyword',
	help='keyword that determines the subset: '+\
		'(i.e. "train", "validate", "test")')

parser.add_option('--percentiles', dest='percentiles', default=None,
    help='percentiles to save from per token embeddings')

parser.add_option('--tsv', '-t', dest='tsv', default=None,
	help='path to the output TSV file')

(options, args) = parser.parse_args()

if(not options.fasta):
	print('%s: input FASTA file is required', file=sys.stderr)

if(not options.pt_dir):
	print('%s: a directory with PT files is required', file=sys.stderr)

options.percentiles = options.percentiles.split(' ')
options.percentiles = list(map(lambda perc: float(perc), options.percentiles))

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

		if('per_tok_representations' not in embs.keys()):
			print('%s protein does not have per_tok_representations' %\
				record.name, file=sys.stderr)
			continue

		embedding = embs['per_tok_representations'][EMB_LAYER]	
		embedding = torch.transpose(embedding, 0, 1)
	
		percentiles =  torch.quantile(embedding[0],
                    torch.tensor(options.percentiles))

		for i in range(1, embedding.size()[0]):
			# Iterating over components
			percentiles = torch.cat((percentiles, torch.quantile(embedding[i], 
				torch.tensor(options.percentiles))))
		
		x.append(percentiles)
		
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

