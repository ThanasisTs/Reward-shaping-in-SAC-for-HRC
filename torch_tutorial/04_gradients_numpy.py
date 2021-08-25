import numpy as np

# f = w * x

# f = 2 * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
	return w * x

# loss = MSE
def loss(y, y_pred):
	return ((y_pred - y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2*x * (w*x - y)
def gradient(x, y, y_pred):
	return np.dot(2*x, y_pred - y).mean()

print('Prediction before training: f(5) = {}'.format(forward(5)))

# Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
	# prediction = forward pass
	y_pred = forward(X)

	# loss
	l = loss(Y, y_pred)

	# gradients
	dw = gradient(X, Y, y_pred)

	# update weights
	w -= dw * learning_rate

	if epoch % 1 == 0:
		print('Epoch: {}: weigths = {}, loss = {}'.format(epoch + 1, w, l))

print("Prediction after training: f(5) = {}".format(forward(5)))