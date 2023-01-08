#!/usr/bin/env python3

# A program that runs the embeddings generation program
# and saves output to a human-readable file

# Prerequisite:
# ESM extract.py program

import sys
import os
from optparse import OptionParser
from data_process import filter_FASTA

parser = OptionParser()

parser.add_option('--fasta', '-f', dest='fasta', default=None,
    help='path to the input FASTA file (required)')

parser.add_option('--no_filter', dest='no_filter', default=False,
    action='store_true', help='setting option to not filter the input')

parser.add_option('--extract', dest='extract',
    default='../programs/esm-0.4.0/extract.py',
    help='path to the embeddings extraction script, default: '+\
        '../programs/esm-0.4.0/extract.py')

parser.add_option('--representations', '-r', dest='representations',
    default='mean', help='set which representations should be extracted. '+\
        'Options: per_tok, mean, std, quantile, hist.')

parser.add_option('--embeddings', '-e', dest='embeddings', default='./emb/',
    help='path to the embeddings directory to save PT files, '+\
        'default: ./emb/')

parser.add_option('--version', '-v', dest='version', action='store_true',
    default=False, help='prints out the version of the program')

(options, args) = parser.parse_args()

if(options.version):
    print(sys.argv[0]+': taken from the program thermoclass version 0.1.0-dev')
    exit()

if(options.fasta == ''):
    print(sys.argv[0]+': required argument -f [input.fasta] is missing',
            file=sys.stderr)
    exit()

print('Filtering input FASTA')
if(options.no_filter):
    filtered_fasta = options.fasta
else:
    # A temporary FASTA file with filtered sequences is generated
    filtered_fasta = filter_FASTA(options.fasta, 1022)

print('Choosing which embeddings to extract')
options.representations = options.representations.split(' ')
print(options.representations)

for representation in options.representations:
    if(representation not in ['per_tok', 'mean', 'std', 'quantile', 'hist']):
        print('%s: representation \'%s\' is unavailable. Skipping.' %\
            (sys.argv[0], representation),
            file=sys.stderr)
        options.representations.remove(representation)

print('Calculating embeddings')
esm_extract_command = 'python3 '+options.extract+' esm1b_t33_650M_UR50S '+\
          filtered_fasta+' '+options.embeddings+' --repr_layers 33 '+\
          '--include '

for representation in options.representations:
    esm_extract_command += representation + ' '

os.system(esm_extract_command)

# Cleaning the environment from the temporary files
if(not options.no_filter):
    os.system('rm %s' % filtered_fasta)

print('Embeddings calculated')
