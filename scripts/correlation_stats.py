#!/usr/bin/env python3.7

# Calculating statistics from the correlation matrix

from optparse import OptionParser
import numpy
import matplotlib.pyplot as plt

parser = OptionParser()

parser.add_option('-i', dest='input',
	help='path to the input file with matrix')

parser.add_option('-t', dest='threshold', default=0.5,
	help='absolute value of the threshold that determines correlation as '+\
	'significant')

parser.add_option('--annot-threshold', dest='annot_threshold', default=10,
	help='set number the threshold for annotation of components with that '+\
	'many high correlation coefficients')

parser.add_option('--hist', dest='hist_file', default=None,
	help='set path to file of output histogram')

parser.add_option('--mean', dest='mean_file', default=None,
	help='set path to file of output mean plot')

parser.add_option('--max', dest='max_file', default=None,
	help='set path to file of output max plot')

parser.add_option('--high-corr', dest='high_corr_file', default=None,
	help='set path to file of output plot with numbers of high correlations')

parser.add_option('--y-max', dest='y_max', default=0,
	help='set the maximum value of y axis')

parser.add_option('--title', dest='title', default=None,
	help='set the title for the plot')

(options, args) = parser.parse_args()

options.threshold = abs(float(options.threshold))
options.annot_threshold = abs(int(options.annot_threshold))
options.y_max = float(options.y_max)

matrix = []

with open(options.input) as f:
	for line in f:
		line = line.strip()
		line = line.replace('[', '')
		line = line.replace(']', '')
		line = line.replace(',', '')

		line = line.split(' ')
		matrix.append(line)

matrix = numpy.array(matrix).astype(float)

max_result = numpy.where(matrix == numpy.amax(matrix))
max_coord_list = list(zip(max_result[0], max_result[1]))
min_result = numpy.where(matrix == numpy.amin(matrix))
min_coord_list = list(zip(min_result[0], min_result[1]))

print('General statistics:')

print('Max: %.3f %s' % (numpy.amax(matrix), str(max_coord_list)))
print('Min: %.3f %s' % (numpy.amin(matrix), str(min_coord_list)))

print('1st quantile: %.3f' % numpy.quantile(matrix, 0.25))
print('2nd quantile: %.3f' % numpy.quantile(matrix, 0.5))
print('3rd quantile: %.3f' % numpy.quantile(matrix, 0.75))

print('Number of elements higher than %.1f: %d' % (options.threshold,
	numpy.sum(abs(matrix) >= options.threshold)))

if(options.hist_file):
	fig = plt.figure(figsize=(6, 5), constrained_layout=True)
	plt.hist(matrix.flatten(), color='navy')
	plt.xlabel('correlation')
	plt.ylabel('number of correlation coefficients')
	if(options.y_max):
		plt.ylim(0, int(options.y_max))
	if(options.title):
		plt.title(options.title)
	plt.savefig(options.hist_file, format='png', dpi=200)
	plt.clf()

if(options.mean_file):
	means = []
	for comp in matrix:
		means.append(numpy.mean(numpy.absolute(comp)))
	means = numpy.array(means)
	plt.plot(numpy.arange(0, len(means)), means, '-', c='silver')
	plt.plot(numpy.arange(0, len(means)), numpy.full((len(means),), numpy.mean(means)), 
		color="navy", label="mean of mean values")
	plt.xlim(-25, len(means)+25)
	plt.xlabel('component')
	plt.ylabel('absolute mean correlation')
	if(options.title):
		plt.title(options.title)
	plt.legend()
	plt.savefig(options.mean_file, format='png', dpi=200)
	plt.clf()

if(options.max_file):
	maxs = [numpy.max(numpy.absolute(comp)) for comp in matrix]
	maxs = numpy.array(maxs)
	mean_max = numpy.mean(maxs)
	plt.plot(numpy.arange(0, len(maxs)), maxs, '-', c='silver')
	plt.plot(numpy.arange(0, len(maxs)), numpy.full((len(maxs,)), 
		mean_max), color="navy", label="mean of maximum values")
	plt.xlabel('component')
	plt.ylabel('absolute maximum correlation')
	if(options.title):
		plt.title(options.title)
	plt.savefig(options.max_file, format='png', dpi=200)
	plt.legend()
	plt.clf()

if(options.high_corr_file):
	high_corrs = []
	for comp in matrix:
		high_corrs.append(numpy.array(numpy.where(abs(comp)
			>= options.threshold)).size)
	high_corrs = numpy.array(high_corrs)
	plt.scatter(numpy.arange(0, len(high_corrs)), high_corrs, c='navy',
		marker='.')
	for i, y in enumerate(high_corrs):
		if(y > options.annot_threshold):
			plt.annotate(str(i), (i,y), textcoords='offset points',
				xytext=(0,-10), ha='center')
	plt.xlabel('component')
	plt.ylabel('number of correlation coefficients higher than '+\
		str(options.threshold))

	yint = []
	locs, labels = plt.yticks()
	for each in locs:
		yint.append(int(each))
	plt.yticks(yint)

	plt.gca().set_ylim(bottom=-1)
	if(options.title):
		plt.title(options.title)
	plt.savefig(options.high_corr_file, format='png', dpi=200)
	plt.clf()
