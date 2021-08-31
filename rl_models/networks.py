import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal

import numpy as np

class CriticNetwork(nn.Module):
	# beta: learning rate
	# input_dims: input dimensions
	# n_actions: number of actions
	# fc1_dims, fc2_dims: number of inputs to the fully connected layers 
	def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
				 name='critic', chkpt_dir='tmp/sac'):
		super(CriticNetwork, self).__init__()

		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

		# The critic evaluates the value of a (state, action) pair, so the input must be the number of states
		# plus the number of actions
		self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.q = nn.Linear(self.fc2_dims, 1)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.to(self.device)

	def forward(self, state, action):
		action_value = self.fc1(torch.cat([state, action], dim=1))
		action_value = F.relu(action_value)
		action_value = self.fc2(action_value)
		action_value = F.relu(action_value)

		q = self.q(action_value)

		return q

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
	# Unlike the CriticNetwork class, we don't need the number of actions here because
	# the value function does not care about the action taken in any state, it only
	# evaluates the states regardless of the action taken 
	def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
				 name='value', chkpt_dir='tmp/sac'):
		super(ValueNetwork, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.v = nn.Linear(self.fc2_dims, 1)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.to(self.device)

	def forward(self, state):
		state_value = self.fc1(state)
		state_value = F.relu(state_value)
		state_value = self.fc2(state_value)
		state_value = F.relu(state_value)

		v = self.v(state_value)
		return v

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
	# The output of the ActorNetwork is a number in the range [0, 1]
	# The real action, however, may need to be scaled. This scale is the max_action param
	def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256,
				 n_actions=2, name='actor', chkpt_dir='tmp/sac'):
		super(ActorNetwork, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = self.fc1_dims
		self.fc2_dims = self.fc2_dims
		self.name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
		self.max_action = max_action
		self.reparam_noise = 1e-6 # is used to avoid calculatin issues like log(0)

		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.mu = nn.Linear(self.fc2_dims, self.n_actions)
		self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
	
	# policy: a probability distibution that tells you the probability of selecting any action
	# from the action space given a state or a set of states
	def sample_normal(self, state, reparameterize=True):
		mu, sigma = self.forward(state)
		probabilities = Normal(mu, sigma)

		if reparameterize:
			actions = probabilities.rsample() # sample + some noise
		else:
			actions = probabilities.sample() # sample without noise

		# set the actions to the [-1, 1] range and then denormalize them by the max_action 
		action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
		
		# the log probabilities are used by the loss function 
		log_probs = probabilities.log_prob(actions)
		log_probs -= torch.log(1-action.pow(2) + self.reparam_noise)

		# scalar quantity for the calculation of the loss
		log_probs = log_probs.sum(1, keepdim=True)

		return action, log_probs

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.checkpoint_file))
	