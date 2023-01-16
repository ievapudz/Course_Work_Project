#!/usr/bin/env python3.7

# A script that extracts principal components of embeddings

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from model_dataset_processing import get_PC_from_ESM
from dataset_processing import filter_sequences
from optparse import OptionParser
import numpy
import re
import random
from Bio import SeqIO

parser = OptionParser()

parser.add_option('--FASTA', dest='fasta', default=None,
                   help='set the path to the FASTA file of the data set')

parser.add_option('--emb-dir', dest='emb_dir', default=None,
                   help='set the path to the directory with embeddings')

parser.add_option('--key', dest='key', default='train',
                   help='determine which subset of the data set should '+\
                        'be used for principal components')

parser.add_option('--npz', '-n', dest='npz', default=None,
                   help='the output NPZ file with principal components')

parser.add_option('--variance', dest='variance', default=None,
                   help='what part of the variance is explained by the '+\
                        'collection of principal components')

parser.add_option('--number', dest='number', default=None,
                   help='choose a number of principal components that '+\
                        ' have to be extracted')

parser.add_option('--threshold', dest='threshold', default=65,
                   help='set temperature threshold')

(options, args) = parser.parse_args()

# Processing options

if(not options.fasta):
    print('%s: the input FASTA file has to be defined.' % sys.argv[0],
        file=sys.stderr)
    exit

if(not options.emb_dir):
    print('%s: the directory with embeddings has to be defined.' % \
        sys.argv[0], file=sys.stderr)
    exit

if(not options.npz):
    print('%s: the output NPZ file has to be defined.' % sys.argv[0],
        file=sys.stderr)
    exit

if(not options.variance and not options.number):
    print('%s: variance to explain or a number of principal components to '+\
        'extract has to be defined.' % sys.argv[0], file=sys.stderr)
    exit

if(options.variance):
    options.variance = float(options.variance)

if(options.number):
    options.number = int(options.number)

if(options.variance and options.number):
    print('%s: both variance and number of PC are given. By default, using '+\
        'the variance to determine the number of principal components.',
        file=sys.stderr)

key = options.key

print("Creating data object")

data = {
    options.key: {
        'X': [],
        'Y': [],
        'FASTA': options.fasta,
        'embeddings': options.emb_dir
    }
}

print("Parsing dataset: "+key)
for record in SeqIO.parse(data[key]['FASTA'], 'fasta'):
    data[key]['X'].append(record)
    seqs_temp = int(record.name.split('|')[2])
    data[key]['Y'].append(seqs_temp)

print("Filtering sequences: "+key)
filter_sequences(data, key, data[key]['embeddings'])

print("Extracting principal components", file=sys.stderr)
pcs = get_PC_from_ESM(data, [key], number_PCs=options.number,
    variance_percent=options.variance, is_read_file_name_id=False)

print("Saving principal components", file=sys.stderr)
pcs_np_arr = numpy.array(pcs)

# Shuffling of the data set
random.seed(27)
temp = list(zip(pcs_np_arr, data[key]['X_filtered'], data[key]['Y_filtered']))
random.shuffle(temp)
pcs_np_arr, data[key]['X_filtered'], data[key]['Y_filtered'] = zip(*temp)

# Extracting information about sequence to save to NPZ
data[key]['Z_filtered'] = []
for record in data[key]['X_filtered']:
    data[key]['Z_filtered'].append(record.name+'|'+str(len(record.seq)))

print('Saving to NPZ')
numpy.savez(options.npz, **{name:value for name, value
    in zip(['x_'+key, 'y_'+key, 'z_'+key],
    [pcs_np_arr, data[key]['Y_filtered'],  data[key]['Z_filtered']])})

print('Saving to TSV')
f = open(re.sub('.npz', '.tsv', re.sub('NPZ', 'TSV', options.npz)), 'w')
for i in range(len(data[key]['Z_filtered'])):
    emb_str = ''
    seq_header = '|'.join(data[key]['Z_filtered'][i].split('|')[0:3])
    for j in range(len(pcs_np_arr[i])):
        emb_str += '%.3f\t' % pcs_np_arr[i][j]

    f.write('%s\t%s\t%s\t%s\t%s\n' %
        (data[key]['Z_filtered'][i].split('|')[0],
        data[key]['Z_filtered'][i].split('|')[1],
        data[key]['Z_filtered'][i].split('|')[3],
        data[key]['Z_filtered'][i].split('|')[2], emb_str))

f.close()
