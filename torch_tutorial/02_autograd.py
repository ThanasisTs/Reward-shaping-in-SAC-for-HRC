import torch
import numpy as np

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)
z = y*y*2
z = z.mean()
print(z)

# gradiant of z wrt x (dz/dx)
z.backward()
print(x.grad)

# Example of autograd

# create weigths
weights = torch.ones(4, requires_grad=True)

# for a number of epochs
for epoch in range(2):
	# simple model
	model_output = (weights*3).sum()

	# backward propagation
	model_output.backward()

	# gradients of the weigths
	print(weights.grad)

	# zero the gradients because in each call .backward they get accumulated
	weights.grad.zero_()