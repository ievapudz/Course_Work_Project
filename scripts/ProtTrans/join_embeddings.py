#!/usr/bin/env python3.7

# Joining two different embeddings

from optparse import OptionParser
from file_actions import load_NPZ_file
import torch
import numpy 

parser = OptionParser()

parser.add_option('--npz-1', dest='npz_1',
				   help='path to the first input NPZ file')

parser.add_option('--npz-2', dest='npz_2',
				   help='path to the second input NPZ file')

parser.add_option('--key', '-k', dest='key',
				   help='the key that denotes the data subset')

parser.add_option('--npz-out', '-o', dest='npz_out',
				   help='path to the output NPZ file')

parser.add_option('--tsv-out', dest='tsv_out',
				   help='path to the output TSV file')

(options, args) = parser.parse_args()

# TODO: error handling if either of NPZ files is not defined

# TODO: error handling if key is not defined

data_1 = load_NPZ_file(options.npz_1, ['x_'+options.key, 'y_'+options.key], 
	['z_'+options.key])

data_2 = load_NPZ_file(options.npz_2, ['x_'+options.key, 'y_'+options.key],
	['z_'+options.key])

x_joined = []

if(options.tsv_out):
	f = open(options.tsv_out, 'w')

for i, record in enumerate(data_1['z_'+options.key]):
	# Joining of embeddings should be done by 'z_'+key values in the NPZ files
	record_idx_2 = list(data_2['z_'+options.key]).index(record)
	joined = torch.cat((data_1['x_'+options.key][i], 
		data_2['x_'+options.key][record_idx_2]))
	
	joined_record_line = '%s\t%s\t%s\t%s\t' % (record.split('|')[0], record.split('|')[1],
		record.split('|')[3], record.split('|')[2])
	for j in range(joined.size(0)):
		joined_record_line += '%.3f\t' % joined[j].tolist()
	if(options.tsv_out):
		print(joined_record_line, file=f)

	joined = joined.numpy()
	x_joined.append(joined)

if(options.tsv_out):
	f.close()

if(options.npz_out):
	numpy.savez(options.npz_out, **{name:value for name,value
		in zip(['x_'+options.key, 'y_'+options.key, 'z_'+options.key], 
		[x_joined, data_1['y_'+options.key], data_1['z_'+options.key]])})
