import torch
import math
from torch import nn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy

def train_epoch_multiclass(model, train_loader, loss_function, optimizer, 
	batch_size, epoch, num_of_classes, pred_file_handle=None):
	"""
	Running training epoch for multiclass classification (can be 
	used for binary classification as well).
	model - PYTORCH MODEL to train
	train_loader - PYTORCH DATALOADER to load the training data
	loss_function - PYTORCH NN module's function for loss evaluation
	optimizer - PYTORCH OPTIM module's optimizer 
	batch_size - INT to determine the size of the mini-batch
	epoch - INT to determine which epoch is currently running
	num_of_classes - INT to determine the number of classes in the 
		classification problem
	pred_file_handle - FILE object opened to write training stage predictions
	"""

	epoch_targets = []
	epoch_outputs = []
	epoch_probabilities = []

	# Iterate over batches
	for i, data in enumerate(train_loader, 0):
		
		inputs, targets = data
		inputs = inputs.to(torch.float32)
		targets = targets.reshape(batch_size, 1)
		targets = targets.to(torch.float32)

		# Zero the gradients
		optimizer.zero_grad()

		# Perform forward pass
		outputs = model(inputs)

		# Compute loss	
		targets = targets.reshape([batch_size])

		loss = loss_function(outputs.float(), targets.long())
		epoch_targets.append(targets)		

		outputs = outputs.detach().numpy()
		output_labels = numpy.array([])

		# Iterating over batch predictions
		for j, class_prob_set in enumerate(outputs):
			# Iterating over class probabilities
			max_ind = 0
			max_el = class_prob_set[max_ind]

			# Finding the class with the highest probability
			for k, el in enumerate(class_prob_set):
				if el > max_el:
					max_ind = k
					max_el = el

			# Collecting batch output labels
			output_labels = numpy.append(output_labels, int(max_ind))

		# Collecting output labels for an epoch
		epoch_outputs.append(output_labels)

		# Collecting probabilities for an epoch
		epoch_probabilities.append(outputs)

		# Perform backward pass
		loss.backward()

		# Perform optimization
		optimizer.step()

	if(pred_file_handle):
		print_epoch_metrics(epoch_outputs, epoch_probabilities, epoch_targets,
			epoch, num_of_epochs=None, ROC_curve_plot_file_prefix=None, 
			pred_file_handle=pred_file_handle, num_of_classes=num_of_classes, 
			flag='T')

def validation_epoch_multiclass(model, validate_loader, loss_function, batch_size,
	epoch, num_of_epochs, ROC_curve_plot_file_prefix, pred_file_handle, 
	num_of_classes=2, validation_flag='V', no_print_probs=False):
	"""
	Running validation epoch for multiclass classification (can be 
	used for binary classification as well).
	model - PYTORCH MODEL to validate
	validate_loader - PYTORCH DATALOADER to load the validation data
	loss_function - PYTORCH NN module's function for loss evaluation
	batch_size - INT to determine the size of the mini-batch
	epoch - INT to determine which epoch is currently running
	num_of_epochs - INT to determine how many epochs will be run
	ROC_curve_plot_file_prefix - STRING that determines the prefix 
		of the ROC curve plot files
	pred_file_handle - FILE object opened to write validation stage predictions
	num_of_classes - INT to determine the number of classes in the 
		classification problem
	validation_flag - STRING to mark the prediction output as validation (i.e. 'V') or
		testing (i.e. 'TE')
	no_print_probs - BOOLEAN flag to set if printing mpredicted class 
		probabilities is not required
	"""

	epoch_targets = []
	epoch_outputs = []
	epoch_probabilities = []

	if(num_of_classes == 2):
		epoch_pos_probs = []

	# Iterate over batches 
	for i, data in enumerate(validate_loader, 0):
		# Get a batch of inputs
		inputs, targets = data
		inputs = inputs.to(torch.float32)
		if(batch_size):
			targets = targets.reshape(batch_size, 1)
		targets = targets.to(torch.float32)

		# Predictions for a batch
		outputs = model(inputs)

		# Measuring loss
		if(batch_size):
			targets = targets.reshape([batch_size])
		loss = loss_function(outputs.float(), targets.long())
		epoch_targets.append(targets)

		outputs = outputs.detach().numpy()	

		output_labels = numpy.array([])

		# Iterating over batch predictions
		for j, class_prob_set in enumerate(outputs):
			# Iterating over class probabilities
			max_ind = 0
			max_el = class_prob_set[max_ind]

			# Finding the class with the highest probability
			for k, el in enumerate(class_prob_set):
				if el > max_el:
					max_ind = k
					max_el = el

			# Collect all probabilities
			probabilities = ''
			for el in class_prob_set:
				probabilities += '%.3f ' % el

			if(num_of_classes == 2):
				epoch_pos_probs.append(class_prob_set[1])

			# Collecting batch output labels
			output_labels = numpy.append(output_labels, int(max_ind))

			if(not no_print_probs):
				# V epoch_idx batch_idx seq_idx_in_batch true_class pred_class probability
				print('%s\t%d\t%d\t%d\t%d\t%d\t%.3f\t%s' %
					(validation_flag, epoch, i, j, int(targets[j].item()), 
					max_ind, max_el, probabilities), file=pred_file_handle)

		# Collecting output labels for an epoch
		epoch_outputs.append(output_labels)

		# Collecting probabilities for an epoch
		epoch_probabilities.append(outputs)
		
	epoch_outputs = numpy.array(epoch_outputs).flatten()
	epoch_probabilities = numpy.array(epoch_probabilities).reshape(
		(len(epoch_outputs), num_of_classes))
	epoch_targets = torch.cat(epoch_targets, dim=0)

	if(num_of_classes > 2):
		print_epoch_metrics(epoch_outputs, epoch_probabilities, epoch_targets,
			epoch, num_of_epochs, ROC_curve_plot_file_prefix, pred_file_handle,
			num_of_classes, validation_flag)
	elif(num_of_classes == 2):
		print_epoch_metrics(epoch_outputs, epoch_probabilities, epoch_targets,
			epoch, num_of_epochs, ROC_curve_plot_file_prefix, pred_file_handle,
			num_of_classes, validation_flag, epoch_pos_probs)

def print_epoch_metrics(epoch_outputs, epoch_probabilities, epoch_targets,
	epoch, num_of_epochs, ROC_curve_plot_file_prefix, pred_file_handle, 
	num_of_classes, flag, epoch_pos_probs=None):
	"""
	Printing metrics for an epoch.
	epoch_outputs - array-like object with output class labels
	epoch_probabilities - array-like object with class probabilities
	epoch_targets - array-like object with true class labels
	epoch - INT that denotes epoch
	num_of_epochs - INT that denotes overall epoch number
	ROC_curve_plot_file_prefix - STRING marking the prefix for ROC files
	pred_file_handle - handle of the opened file for writing predictions
	num_of_classes - INT that denotes the number of classes
	flag - STRING that denotes the stage
	"""

	MCC_epoch = metrics.matthews_corrcoef(epoch_targets, epoch_outputs)
	accuracy_epoch = metrics.accuracy_score(epoch_targets, epoch_outputs)
	loss_epoch = metrics.log_loss(epoch_targets, epoch_probabilities, 
		labels=[float(j) for j in range(num_of_classes)])

	if(flag == 'V' or flag == 'TE'):
		epoch_mark = '# Epoch'
	elif(flag == 'T'):
		epoch_mark = '# T_Epoch'
	
	if(num_of_classes == 2):
		# Processing binary case
		precision_epoch = metrics.precision_score(epoch_targets, epoch_outputs)
		recall_epoch = metrics.recall_score(epoch_targets, epoch_outputs)
		ROC_AUC_epoch = metrics.roc_auc_score(epoch_targets, epoch_pos_probs) 

		print(epoch_mark+', MCC, accuracy, loss, precision, recall, ROC_AUC: '+
			'%d %.3f %.3f %.3f %.3f %.3f %.3f' %
			(epoch, MCC_epoch, accuracy_epoch, loss_epoch, precision_epoch,
			recall_epoch, ROC_AUC_epoch), file=pred_file_handle)
		if(ROC_curve_plot_file_prefix):
			if(flag == 'TE'):
				plot_ROC_curve(epoch_targets, num_of_epochs, epoch_pos_probs, 
					ROC_curve_plot_file_prefix+'_'+str(epoch)+'.png', True)
			else:
				plot_ROC_curve(epoch_targets, num_of_epochs, epoch_pos_probs,
					ROC_curve_plot_file_prefix+'_'+str(epoch)+'.png')
	else:
		print(epoch_mark+', MCC, accuracy, loss: %d %.3f %.3f %.3f' % 
			(epoch, MCC_epoch, accuracy_epoch, loss_epoch), 
			file=pred_file_handle)

	print('# Confusion matrix ', file=pred_file_handle)
	conf_m = confusion_matrix(epoch_targets, epoch_outputs, 
			 labels=range(0, num_of_classes))
	for i in range(num_of_classes):
		print('# '+str(conf_m[i]), file=pred_file_handle)

def plot_ROC_curve(targets, num_of_epochs, outputs, fig_name, clear_plot=False):
	"""
	Plotting the ROC curve.
	targets - array-like object to store the true values
	num_of_epochs - INT noting how many epochs were executed
	outputs - array-like object to store the prediction values (raw)
	fig_name - STRING to set the name for the plot
	clear_plot - BOOLEAN that determines whether to clear the plot 
		before plotting or not
	"""
	if clear_plot:
		plt.clf()
	fpr, tpr, _ = metrics.roc_curve(targets, outputs)
	iterated_colors = [plt.get_cmap('jet')(1. * i/num_of_epochs) for i in range(num_of_epochs)]
	mpl.rcParams['axes.prop_cycle'] = cycler('color', iterated_colors)
	plt.plot(fpr,tpr)
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(fig_name)
	
# A function that makes inferences about unlabelled data
def inference_epoch(model, test_loader, threshold, file_for_predictions='', 
	labelled=False, binary_predictions_only=True, identifiers=[]):
	"""
	Making inferences about the unlabelled data.
	model - PYTORCH MODEL to test
	test_loader - PYTORCH DATALOADER to load the inference data
	threshold - INT for temperature threshold setting
	file_for_predictions - STRING to the path of the file with inferences
	labelled - BOOLEAN to set if the data set was actually labelled
	binary_predictions_only - BOOLEAN to set if only binary prediction labels 
		are needed to print
	identifiers - LIST of identifiers in the inference data set
	"""
	
	epoch_outputs = []

	if(file_for_predictions != ''):
		file_handle = open(file_for_predictions, 'w')

	if(len(identifiers)):
		identifier_header = 'protein\t'
	else:
		identifier_header = ''

	if(labelled):
		file_handle.write(identifier_header+"temperature\ttrue_label\t"+\
			"predicted_label\tprediction\n")
	else:
		file_handle.write(identifier_header+"predicted_label\tprediction\n")

	# Iterate over the DataLoader for testing data
	for i, data in enumerate(test_loader, 0):
		inputs, targets = data
		outputs = model(inputs.float())
		outputs = outputs.detach().numpy()
		epoch_outputs.append(outputs)
	
		# Printing prediction values
		for j, output in enumerate(outputs):
			if(len(identifiers)):
				file_handle.write("%s\t" % identifiers[i].split('|')[1])
			if(labelled):
				file_handle.write("%d\t%d\t" % (targets[i].item(), 
					int(targets[i].item()/65)))
			if(binary_predictions_only):
				if(output[1] >= threshold):
					file_handle.write("1\n")
				else:
					file_handle.write("0\n")
			else:
				if(output[1] >= threshold):
					file_handle.write("1\t"+str(output[1])+"\n")
				else:
					file_handle.write("0\t"+str(output[1])+"\n")

	file_handle.close()


