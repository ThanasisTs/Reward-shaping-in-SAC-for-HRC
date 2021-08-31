import os
import torch
import torch.nn.functional as F

import numpy as np

from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
	def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
				 env=None, gamma=0.99):