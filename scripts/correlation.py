#!/usr/bin/env python3.7

# Calculation of correlation coefficients between components

from optparse import OptionParser
from file_actions import load_NPZ_file
import torch
import numpy
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = OptionParser()

parser.add_option('--npz', dest='npz',
    help='path to the input NPZ file')

parser.add_option('--key', '-k', dest='key',
    help='the key that denotes the data subset')

parser.add_option('--boundary', '-b', dest='boundary',
    help='the number that separated components')

parser.add_option('--png-out', dest='png_out',
    help='path to the output PNG file')

parser.add_option('--matrix-out', dest='matrix_out',
    help='path to the output file with correlation matrix')

(options, args) = parser.parse_args()

data = load_NPZ_file(options.npz, ['x_'+options.key, 'y_'+options.key],
    ['z_'+options.key])

corr_matrix = []

for i in tqdm(range(int(options.boundary))):
    ith_components = []
    ith_correlations = []
    for j in range(len(data['x_'+options.key])):
        ith_components.append(data['x_'+options.key][j][i].item())

    for k in tqdm(range(int(options.boundary), len(data['x_'+options.key][0]))):
        kth_components = []
        for j in range(len(data['x_'+options.key])):
            kth_components.append(data['x_'+options.key][j][k].item())

        ith_correlations.append(numpy.corrcoef(ith_components,
            kth_components)[0][1])

    corr_matrix.append(ith_correlations)

matrix_file_handle = open(options.matrix_out, 'w')

for i in range(len(corr_matrix)):
    print(corr_matrix[i], file=matrix_file_handle)

matrix_file_handle.close()

plt.imshow(numpy.array(corr_matrix), cmap='bwr', aspect='auto')
plt.xlabel(str(int(options.boundary))+'-'+\
    str(len(data['x_'+options.key][0])-1)+' components')
plt.ylabel('0-'+str(int(options.boundary)-1)+' components')
plt.colorbar()
plt.savefig(options.png_out, format='png')


