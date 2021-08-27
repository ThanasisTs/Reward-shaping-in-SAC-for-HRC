'''
1) Design model (input, output, forward pass)
2) Construct loss and optimizer
3) Training loop
	- forward pass: compute prediction and loss
	- backward pass: gradients
	- update weights
'''

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_train = y_train.view(y_train.shape[0], 1)
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_test = y_test.view(y_test.shape[0], 1)


# 1) model
class LogisticRegression(nn.Module):
	def __init__(self, n_input_features):
		super().__init__()
		self.linear = nn.Linear(n_input_features, 1)

	def forward(self, x):
		return torch.sigmoid(self.linear(x))

model = LogisticRegression(n_features)


# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
n_epochs = 1000
for epoch in range(n_epochs):
	# forward pass
	y_pred = model(X_train)
	loss = criterion(y_pred, y_train)

	# backward pass
	loss.backward()

	# update weights
	optimizer.step()
	optimizer.zero_grad()

	if (epoch + 1) % 10 == 0:
		print(f"Epoch {epoch+1}: loss = {loss:.4f}")

with torch.no_grad():
	y_pred = model(X_test)
	y_pred_cls = y_pred.round()
	print(type(y_pred_cls))
	acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
	print(f'Accuracy = {acc:.4f}')