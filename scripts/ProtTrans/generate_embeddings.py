#!/usr/bin/env python3.7

# A script that generates ProtTrans embeddings for a given
# FASTA file

# Example usage:
# ./scripts/ProtTrans/generate_embeddings.py --fasta thermoclass/FASTA/1ceu.fasta

import sys
import os
from os.path import exists
from optparse import OptionParser
from prottrans_models import get_pretrained_model
from prottrans_models import get_tokenizer
from prottrans_models import save_pretrained_model
from prottrans_models import process_FASTA
from prottrans_models import get_embeddings
from datetime import datetime
import numpy
import torch

parser = OptionParser()

parser.add_option('--fasta', '-f', dest='fasta',
    help='path to the input FASTA file')

parser.add_option('--emb_dir', '-d', dest='embeddings_dir', default='./',
    help='the name of the directory for embeddings')

parser.add_option('--representations', '-r', dest='representations',
    default='mean', help='choose types of represetations to save: '+\
    'per_res, mean, std, quantile, hist')

parser.add_option('--model', dest='model', default=None,
    help='set path to the location for the downloaded model')

parser.add_option('--version', '-v', dest='version', action='store_true',
    default=False, help='prints out the version of the program')

(options, args) = parser.parse_args()

options.representations = options.representations.split(' ')

if(options.version):
    print(sys.argv[0]+': version 0.1.0-dev')
    exit()

MODEL_PATH = 'Rostlab/prot_t5_xl_half_uniref50-enc'
FASTA_FILE = options.fasta

assert FASTA_FILE != None
assert options.model != None

# Loading or downloading the model
now = datetime.now()
print("Beginning to load the model: %s" % now)

if(os.path.isfile(options.model+'/pytorch_model.bin')):
    model = get_pretrained_model(options.model)
else:
    model = get_pretrained_model(MODEL_PATH)
    save_pretrained_model(model, options.model)

now = datetime.now()
print("Finished loading the model: %s" % now)

now = datetime.now()
print("Beginning to load the tokenizer: %s" % now)
tokenizer = get_tokenizer(MODEL_PATH)

now = datetime.now()
print("Finishing to load the tokenizer: %s" % now)

seqs = process_FASTA(FASTA_FILE)

results = get_embeddings(model, tokenizer, seqs,
    'per_res' in options.representations,
    'mean' in options.representations,
    'std' in options.representations,
    'quantile' in options.representations,
    'hist' in options.representations)

for seq in list(seqs.keys()):
    results_seq = {'label': seq}
    for key in options.representations:
        results_seq[key+'_representations'] = torch.from_numpy(
            results[key+'_representations'][seq])
    torch.save(results_seq, options.embeddings_dir+'/'+seq+'.pt')
