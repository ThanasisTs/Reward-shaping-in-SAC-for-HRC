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

# f = w * x

# f = 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
	return w * x

# loss = MSE
def loss(y, y_pred):
	return ((y_pred - y)**2).mean()

print('Prediction before training: f(5) = {}'.format(forward(5)))

# Training
learning_rate = 0.01
n_iters = 1000

for epoch in range(n_iters):
	# prediction = forward pass
	y_pred = forward(X)

	# loss
	l = loss(Y, y_pred)

	# gradients = backward pass
	l.backward()

	# update weights
	with torch.no_grad():
		w -= w.grad * learning_rate
		w.grad.zero_()

	if epoch % 10 == 0:
		print('Epoch: {}: weigths = {}, loss = {}'.format(epoch + 1, w, l))

print("Prediction after training: f(5) = {}".format(forward(5)))