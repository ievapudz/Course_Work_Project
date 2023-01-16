from torch import nn

class SLP_with_sigmoid(nn.Module):
	def __init__(self, input_size=None):

		if(input_size == None):
			self.input_size = 1280
		else:
			self.input_size = int(input_size)

		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(self.input_size, 2),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.layers(x)

