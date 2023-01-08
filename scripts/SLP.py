from torch import nn

class SLP(nn.Module):

	# This model requires BCEWithLogitsLoss function to be used in the 
	# model's flow.
	
	def __init__(self, input_size=None):

		if(input_size == None):
			self.input_size = 1280
		else:
			self.input_size = int(input_size)

		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(self.input_size, 1),
		)

	def forward(self, x):
		return self.layers(x)

class SLP_with_sigmoid(nn.Module):

	# This model requires BCELoss function to be used in the model's flow.

	def __init__(self, input_size=None):

		if(input_size == None):
			self.input_size = 1280
		else:
			self.input_size = int(input_size)

		super().__init__()
		self.layers = nn.Sequential(
			# DEBUG
			nn.Linear(self.input_size, 2),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.layers(x)

