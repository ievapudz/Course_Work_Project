#!/usr/bin/env python3.7

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import numpy
from optparse import OptionParser
from torch import nn
from model_dataset_processing import load_tensor_from_NPZ
from model_dataset_processing import trim_dataset
from model_dataset_processing import convert_labels_to_binary
from SLP import SLP_with_sigmoid
from MLP import MLP_C2H1
from MLP import MLP_C2H2
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from model_flow import train_epoch_multiclass
from model_flow import validation_epoch_multiclass
from sklearn.utils import class_weight
from torch.utils.data import WeightedRandomSampler

parser = OptionParser()
parser.add_option("--npz-train", dest="npz_train",
	help="path to the NPZ file with the training set")

parser.add_option("--npz-validate", dest="npz_validate",
	help="path to the NPZ file with the validation set")

parser.add_option("--npz-test", dest="npz_test",
	help="path to the NPZ file with testing set")

parser.add_option("--no-convert", dest="convert",
	action="store_false", default=True, help="option to skip "+\
		"or execute temperature labels conversion to class labels")

parser.add_option("--architecture", "-a", dest="architecture",
	help="choose the model's architecture to train "+\
		"by giving the name of the model (i.e. SLP)")

parser.add_option("--hidden", dest="hidden", default=None,
	help="setting sizes of the hidden layers if the architecture is MLP")

parser.add_option("--batch", "-b", dest="batch_size",
	default=24, help="batch size")

parser.add_option("--learning_rate", "-l", dest="learning_rate",
	default=1e-4, help="learning rate")

parser.add_option("--epochs", "-e", dest="epochs",
	default=5, help="number of epochs")

parser.add_option("--model", "-m", dest="model_dir",
	default=None, help="directory to save the model file")

parser.add_option("--ROC_dir", "-r", dest="ROC_dir",
	help="directory with ROC curves")

parser.add_option("--input-size", "-i", dest="input_size",
	default=None, help="set the size of input vectors")

parser.add_option("--pred-dir", "-p", dest="predictions_dir",
	help="directory for the output TSV file with predictions")

parser.add_option("--weighted", "-w", dest="weighted", action="store_true", 
	default=False, help="use weighted cross entropy loss function for "+\
		"imbalanced data sets")

parser.add_option("--threshold", dest="threshold", default=65,
    help="setting the threshold to set up the binary labels")

(options, args) = parser.parse_args()

# Setting hyperparameters
BATCH_SIZE = int(options.batch_size)
NUM_OF_EPOCHS = int(options.epochs)
LEARNING_RATE = float(options.learning_rate)

if(options.hidden):
	options.hidden = options.hidden.split(' ')

if(options.npz_train and options.npz_validate):
	# Processing the data set

	# The shuffling of data sets happens in embeddings saving script
	train_npz_dataset = load_tensor_from_NPZ(options.npz_train, ['x_train', 'y_train'])
	validate_npz_dataset = load_tensor_from_NPZ(options.npz_validate, 
		['x_validate', 'y_validate'])

	trim_dataset(train_npz_dataset, ['x_train', 'y_train'], BATCH_SIZE)
	trim_dataset(validate_npz_dataset, ['x_validate', 'y_validate'], BATCH_SIZE)

	if(options.convert):
		convert_labels_to_binary(train_npz_dataset, ['y_train'], 
			threshold=int(options.threshold))
		convert_labels_to_binary(validate_npz_dataset, ['y_validate'], 
			threshold=int(options.threshold))

	train_dataset = TensorDataset(train_npz_dataset['x_train'], 
		train_npz_dataset['y_train'])  

	validate_dataset = TensorDataset(validate_npz_dataset['x_validate'], 
		validate_npz_dataset['y_validate'])

	if(options.weighted):
		train_samples_weights = class_weight.compute_sample_weight('balanced',
			train_dataset[0:][1])

		train_sampler = WeightedRandomSampler(train_samples_weights,
			len(train_samples_weights))

		train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
			sampler=train_sampler)

		validate_samples_weights = class_weight.compute_sample_weight('balanced',
			validate_dataset[0:][1])

		validate_sampler = WeightedRandomSampler(validate_samples_weights,
			len(validate_samples_weights))

		validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE,
			sampler=validate_sampler)
	else:
		train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
			shuffle=False)
		validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE,
			shuffle=False)

# Set fixed random number seed
torch.manual_seed(42)

MODEL_PATH = ''
ROC_PATH = ''

# Initialize the model architecture
if(options.architecture == 'SLP'):
	model = SLP_with_sigmoid(options.input_size)
elif(options.architecture == 'MLP_C2H1'):
	model = MLP_C2H1(options.input_size, options.hidden[0])
elif(options.architecture == 'MLP_C2H2'):
	if(len(options.hidden) < 2):
		print('%s: too few hidden layer sizes were set' % (sys.argv[0]), 
			file=sys.stderr)
		exit
	model = MLP_C2H2(options.input_size, options.hidden[0], options.hidden[1])

if(options.model_dir):
	MODEL_PATH = '%s/model_l%.0e_b%d_e%d.pt' % (options.model_dir, 
		LEARNING_RATE, BATCH_SIZE, NUM_OF_EPOCHS)

else:
	print(sys.argv[0]+': the destination directory of file for the model '+\
		'required. Provide option -m.', file=sys.stderr)
	exit

# Preparing saving of predictions
predictions_file = ''

if(options.predictions_dir):

	if(not os.path.exists(options.predictions_dir)):
		os.makedirs(options.predictions_dir)

	predictions_file = '%s/predictions_l%.0e_b%d_e%d.tsv' % \
		(options.predictions_dir, LEARNING_RATE, BATCH_SIZE, NUM_OF_EPOCHS)

else:
	print(sys.argv[0]+': the destination directory of the predictions '+\
		'file is required.', file=sys.stderr)

if(options.weighted):
	weights = class_weight.compute_class_weight('balanced',
			classes=numpy.unique(train_dataset[0:][1]),
			y=train_dataset[0:][1].numpy())
	weights_list = torch.tensor(weights,dtype=torch.float)

	weights = torch.tensor([weights_list[0],
							weights_list[1]])

	loss_function = nn.CrossEntropyLoss(weight=weights)
else:
	loss_function = nn.CrossEntropyLoss()

# Setting loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Preparing to write predictions to the file
pred_file_handle = open(predictions_file, 'w')

if(options.npz_train and options.npz_validate):
	# Preparing ROC curve saving
	ROC_prefix_v = ''

	if(options.ROC_dir):
		if(not os.path.exists(options.ROC_dir)):
			os.makedirs(options.ROC_dir)
		
		ROC_PATH = options.ROC_dir
		ROC_prefix_v = '%s/v_l%.0e_b%d_e%d' % (ROC_PATH, LEARNING_RATE,
			BATCH_SIZE, NUM_OF_EPOCHS)

	# Training and validation
	for epoch in range(0, NUM_OF_EPOCHS):
		train_epoch_multiclass(model, train_loader, loss_function, optimizer, 
					BATCH_SIZE, epoch, num_of_classes=2)
		validation_epoch_multiclass(model, validate_loader, loss_function, 
					BATCH_SIZE, epoch, NUM_OF_EPOCHS, ROC_prefix_v, pred_file_handle,
					num_of_classes=2)

	torch.save(model.state_dict(), MODEL_PATH)

# Testing stage
if(options.npz_test):
	test_npz_dataset = load_tensor_from_NPZ(options.npz_test, 
		['x_test', 'y_test'])

	if(options.convert):
		convert_labels_to_binary(test_npz_dataset, ['y_test'], 
			threshold=int(options.threshold))
	
	test_dataset = TensorDataset(test_npz_dataset['x_test'], 
		test_npz_dataset['y_test'])
	test_loader = DataLoader(test_dataset, shuffle=False)

	if(options.ROC_dir):
		if(not os.path.exists(options.ROC_dir)):
			os.makedirs(options.ROC_dir)

		ROC_PATH = options.ROC_dir
		ROC_prefix_te = ROC_PATH + 'te_l' + \
			numpy.format_float_scientific(LEARNING_RATE, precision=0,
				exp_digits=1) + '_b' + str(BATCH_SIZE) + '_e' + \
				str(NUM_OF_EPOCHS)

		ROC_prefix_te = '%s/te_l%.0e_b%d_e%d' % (ROC_PATH, LEARNING_RATE,
			BATCH_SIZE, NUM_OF_EPOCHS)

	model.load_state_dict(torch.load(MODEL_PATH))
	model.eval()

	validation_epoch_multiclass(model, test_loader, loss_function,
		batch_size=None, epoch=0, num_of_epochs=1, 
		ROC_curve_plot_file_prefix=ROC_prefix_te, 
		pred_file_handle=pred_file_handle, num_of_classes=2,
		validation_flag='TE')

# Saving prediction results
pred_file_handle.close()

