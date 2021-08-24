import torch
import numpy as np

# torch.empty() creates an empty tensor of specified size
x = torch.empty(2, 3)
print(x)

# torch.rand() creates a tensor of specified size with random values
x = torch.rand(2, 2)
print(x)

# torch.zeros()/ones() creates a tensor of specified size with zero/one values
x = torch.zeros(2, 3)
print(x)

# suppose x is a torch tensor
# x.dtype returns the type of the tensor
print(x.dtype)

# we can specify the type of the tensor
x = torch.ones(2, 2, dtype=torch.int)
print(x.dtype)

# size of a tensor
print(x.size())

# construct a tensor from list
x = torch.tensor([1, 2, 3, 5])
print(x)

# basic operations
x = torch.rand(2, 2)
y = torch.rand(2, 2)

# element-wise addition
z = x + y
z = torch.add(x, y)
y.add_(x) # y = x + y (in torch, any function with a trailing underscore will perform an in-place operation)

# element-wise substraction
z = x - y 
z = torch.sub(x, y)

# element-wise multiplication
z = x * y
z = torch.mul(x, y)

# element-wise division
z = x / y
z = torch.div(x, y)

# slicing
x = torch.rand(5, 3)
print(x[:, 0]) # first column, all the rows
print(x[1, 1]) # (1,1) element
print(x[1, 1].item()) # gets the actual value of the tensor, works only in tensors with one elem

# reshaping (-1 works the same as in numpy.reshape())
x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)

# numpy to torch tensor
# SOS: if the tensors are stored in the CPU, then the torch tensor and the respective
# numpy array share the same memory location, meaning that modifying the one changes
# the other
x = torch.ones(5)
print(x)
y = x.numpy()
print(type(y))

# torch tensor to numpy
x = np.ones(5)
print(x)
y = torch.from_numpy(x)
print(y)

# check if GPU is available
if torch.cuda.is_available():
	print('GPU')
	device = torch.device("cuda")
	x = torch.ones(5, device=device)
	x = x.to("cpu")
	print('ok')
else:
	print('CPU')

