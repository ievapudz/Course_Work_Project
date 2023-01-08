import numpy
import torch
import sys
import os

# A function that loads embeddings to dictionary of numpy arrays with according keywords
# Example keywords: ['x_train', 'y_train']
def load_numpy_from_NPZ(NPZ_file, keywords):
	dataset = {}
	with numpy.load(NPZ_file) as data_loaded:
		for i in range(len(keywords)):
			dataset[keywords[i]] = data_loaded[keywords[i]]
	return dataset

def load_NPZ_file(NPZ_file, keywords_torch, keywords_other=None):
	'''
	Loading embeddings from an NPZ file.
	NPZ_file - STRING that identifies the file to read
	keywords - LIST of keywords that identify what to extract from an NPZ
	'''
	dataset = {}
	with numpy.load(NPZ_file, allow_pickle=True) as data_loaded:
		for i in range(len(keywords_torch)):
			dataset[keywords_torch[i]] = torch.from_numpy(
				data_loaded[keywords_torch[i]])
		if(keywords_other):
			for i in range(len(keywords_other)):
				dataset[keywords_other[i]] = data_loaded[keywords_other[i]]
	return dataset

# A function that loads embeddings to dictionary with according keywords
# Example keywords: ['x_train', 'y_train']
def load_tensor_from_NPZ(NPZ_file, keywords):
	dataset = {}
	with numpy.load(NPZ_file, allow_pickle=True) as data_loaded:
		for i in range(len(keywords)):
			dataset[keywords[i]] = torch.from_numpy(data_loaded[keywords[i]])
	return dataset

# A function that trims the dataset so that its length would divide from the number of batches
def trim_dataset(dataset, keywords, batch_size):
	for i in range(len(keywords)):
		residual = len(dataset[keywords[i]]) % batch_size
		if(residual != 0):
			dataset[keywords[i]] = dataset[keywords[i]][0:len(dataset[keywords[i]])-residual]

# A function that converts non-binary labels to binary
def convert_labels_to_binary(dataset, keywords, threshold=65):
	for keyword in keywords:
		for i in range(len(dataset[keyword])):
			if(dataset[keyword][i].item() >= threshold):
				dataset[keyword][i] = 1
			elif(dataset[keyword][i].item() < threshold):
				dataset[keyword][i] = 0

# Conversion of temperature to an appropriate class label
def convert_labels_to_temperature_class(dataset, keywords, ranges=None):
	if(not ranges):
		beg = numpy.array([0])
		middle = numpy.arange(15, 86, 5)
		end = numpy.array([100])
		ranges = numpy.concatenate((beg, middle, end), axis=None)
	for keyword in keywords:
		for i in range(len(dataset[keyword])):
			for j, r in enumerate(ranges):
				if(j+1 < len(ranges)):
					if(dataset[keyword][i].item() >= ranges[j] and dataset[keyword][i].item() < ranges[j+1]):
						dataset[keyword][i] = j
						continue
					#elif(dataset[keyword][i].item() >= ranges[-1]):
					#	dataset[keyword][i] = j-1
					#	continue

# A function that normalises temperature labels
def normalise_labels(dataset, keywords, denominator):
	for keyword in keywords:
		normalised_labels = []
		for i in range(len(dataset[keyword])):
			float_tensor_normalised = torch.tensor(float(dataset[keyword][i].item() / denominator), dtype=torch.float32)
			normalised_labels.append(float_tensor_normalised)
		dataset[keyword] = torch.FloatTensor(normalised_labels)

# A function that normalises temperature labels to z-scores
def normalise_labels_as_z_scores(dataset, keywords, ref_point):
	# ref_point - by default, it is mean of sample to calculate z-score. 
	#			 This function allows to calculate z-score from point other
	#			 than the mean. 
	for keyword in keywords:
		normalised_labels = []
		std = standard_deviation(dataset[keyword], ref_point)
		for i in range(len(dataset[keyword])):
			float_tensor_normalised = torch.tensor(float((dataset[keyword][i].item() - ref_point) / std), dtype=torch.float32)
			normalised_labels.append(float_tensor_normalised)
		dataset[keyword] = torch.FloatTensor(normalised_labels)

# Conversion of z-score normalisation back to a temperature label
def convert_z_score_to_label(z_score_tensor, ref_point, std):
	# ref_point - by default, it is mean of sample to calculate z-score. 
	#			 This function allows to calculate z-score from point other
	#			 than the mean. 
	return torch.tensor(float(z_score_tensor.item() * std + ref_point), dtype=torch.float32)

# A function to calculate standard deviation
def standard_deviation(dataset, mean):
	N = len(dataset)
	var = 0
	for i in range(N):
		var += (float(dataset[i].item() - mean))**2
	var *= float(1/N)
	return var**(0.5)
