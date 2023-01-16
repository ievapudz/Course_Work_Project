import numpy
import torch
import sys
import os

def load_numpy_from_NPZ(NPZ_file, keywords):
	"""
	A function that loads embeddings to dictionary of numpy 
	arrays with according keywords.
	NPZ_file - STRING that determines a path to the NPZ file
	keywords - LIST with keywords of the inner NPZ dictionary
		Example keywords: ['x_train', 'y_train']
	returns the data set as DICT
	"""
	dataset = {}
	with numpy.load(NPZ_file) as data_loaded:
		for i in range(len(keywords)):
			dataset[keywords[i]] = data_loaded[keywords[i]]
	return dataset

def load_NPZ_file(NPZ_file, keywords_torch, keywords_other=None):
	"""
	Loading embeddings from an NPZ file.
	NPZ_file - STRING that determines a path to the NPZ file
	keywords - LIST of keywords that identify what to extract from an NPZ
	returns the data set as DICT
	"""
	dataset = {}
	with numpy.load(NPZ_file, allow_pickle=True) as data_loaded:
		for i in range(len(keywords_torch)):
			dataset[keywords_torch[i]] = torch.from_numpy(
				data_loaded[keywords_torch[i]])
		if(keywords_other):
			for i in range(len(keywords_other)):
				dataset[keywords_other[i]] = data_loaded[keywords_other[i]]
	return dataset

def load_tensor_from_NPZ(NPZ_file, keywords):
	"""
	A function that loads embeddings to dictionary 
	with according keywords.
	NPZ_file - STRING that determines a path to the NPZ file
	keywords - LIST of keywords that identify what to extract from an NPZ
	returns the data set as DICT
	"""
	dataset = {}
	with numpy.load(NPZ_file, allow_pickle=True) as data_loaded:
		for i in range(len(keywords)):
			dataset[keywords[i]] = torch.from_numpy(data_loaded[keywords[i]])
	return dataset

def trim_dataset(dataset, keywords, batch_size):
	"""
	A function that trims the dataset so that its length would divide 
	from the number of batches.
	dataset - DICT of the data set
	keywords - LIST of keys of the dictionary to take into account
	batch_size - INT of the size of mini-batch for training
	"""
	for i in range(len(keywords)):
		residual = len(dataset[keywords[i]]) % batch_size
		if(residual != 0):
			dataset[keywords[i]] = dataset[keywords[i]][0:len(dataset[keywords[i]])-residual]

def convert_labels_to_binary(dataset, keywords, threshold=65):
	"""
	A function that converts non-binary labels to binary.
	dataset - DICT of the data set
	keywords - LIST of keys of the dictionary to take into account
	threshold - FLOAT that represents the temperature threshold to 
		distinguish between thermostable and not thermostable proteins
	"""
	for keyword in keywords:
		for i in range(len(dataset[keyword])):
			if(dataset[keyword][i].item() >= threshold):
				dataset[keyword][i] = 1
			elif(dataset[keyword][i].item() < threshold):
				dataset[keyword][i] = 0

def normalise_labels_as_z_scores(dataset, keywords, ref_point):
	"""
	Normalisation of temperature labels to z-scores.
	dataset - DICT of the data set
	keywords - LIST of keys of the dictionary to take into account
	ref_point - FLOAT, which by default, it should be mean of the 
		sample to calculate z-score. 
	"""
	for keyword in keywords:
		normalised_labels = []
		std = standard_deviation(dataset[keyword], ref_point)
		for i in range(len(dataset[keyword])):
			float_tensor_normalised = torch.tensor(float((dataset[keyword][i].item() - ref_point) / std), dtype=torch.float32)
			normalised_labels.append(float_tensor_normalised)
		dataset[keyword] = torch.FloatTensor(normalised_labels)

def standard_deviation(dataset, mean):
	"""
	Helper function to calculate standard deviation.
	dataset - DICT of the data set
	mean - FLOAT representing the mean of the sample
	returns calculated standard deviation as FLOAT 
	"""
	N = len(dataset)
	var = 0
	for i in range(N):
		var += (float(dataset[i].item() - mean))**2
	var *= float(1/N)
	return var**(0.5)
