#!/usr/bin/env python3.7

from optparse import OptionParser
import numpy

parser = OptionParser()

parser.add_option('--tsv', '-t', dest='tsv',
                   help='path to the input TSV file')

parser.add_option('--key', '-k', dest='key',
                   help='the key that denotes the data subset')

parser.add_option('--npz', '-n', dest='npz',
                   help='path to the destination NPZ file')

(options, args) = parser.parse_args()

lines = []
with open(options.tsv) as f:
    lines = f.readlines()

x = []
y = []
z = []
for line in lines:
    line= line.strip()
    z_el = '%s|%s|%s|%s' % (line.split('\t')[0], line.split('\t')[1],
        line.split('\t')[3], line.split('\t')[2])
    z.append(z_el)
    x_el = line.split('\t')[4:]
    x_el = [float(x_el_el) for x_el_el in x_el]
    x.append(x_el)
    y.append(int(line.split('\t')[3]))

numpy.savez(options.npz, **{name:value for name, value
    in zip(['x_'+options.key, 'y_'+options.key, 'z_'+options.key],
    [x, y, z])})
