'''
Steps
	1) Design model (input, output size, forward pass)
	2) Construct loss and optimizer
	3) Training loop
		- forward pass
		- backward pass
		- update weights
'''

import torch
import torch.nn as nn

# f = w * x

# f = 2 * x

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(LinearRegression, self).__init__()
		# define layers
		self.lin = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return self.lin(x)


model = LinearRegression(input_size, output_size)


print('Prediction before training: f(5) = {}'.format(model(X_test).item()))

# Training
learning_rate = 0.01
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
	# prediction = forward pass
	y_pred = model(X)

	# loss
	l = loss(Y, y_pred)

	# gradients = backward pass
	l.backward()

	# update weights
	optimizer.step()
	optimizer.zero_grad()

	if epoch % 10 == 0:
		[w, b] = model.parameters()
		print('Epoch: {}: weigths = {}, loss = {}'.format(epoch + 1, w[0][0].item(), l))

print("Prediction after training: f(5) = {}".format(model(X_test).item()))