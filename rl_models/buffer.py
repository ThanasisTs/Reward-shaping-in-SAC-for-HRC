import numpy as np

class ReplayBuffer():
	# max_size: maximum size of memory
	# input_shape: observation dimensionality of the environment
	# n_actions: we will be dealing with continuous action space
	def __init__(self, max_size, input_shape, n_actions):
		self.mem_size = max_size
		self.mem_cntr = 0 # memory counter, keeps track of the position of the first available memory
		self.state_memory = np.zeros((self.mem_size, *input_shape)) # memory initialization
		self.new_state_memory = np.zeros((self.mem_size, *input_shape)) # keep track of the state after an action
		self.action_memory = np.zeros((self.mem_size, n_actions)) # memory of the actions
		self.reward_memory = np.zeros(self.mem_size) # memory of the rewards
		self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

	# store a new transition
	def store_transition(self, state, action, reward, state_, done):
		# figure out where I should store the transition in the memory
		index = self.mem_cntr % self.mem_size

		# store the data
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.terminal_memory[index] = done

		self.mem_cntr += 1


	def sample_buffer(self, batch_size):
		# number of available transitions in memory
		max_mem = min(self.mem_cntr, self.mem_size)

		# batch = random samples
		batch = np.random.choice(max_mem, batch_size)

		states = self.state_memory[batch]
		states_ = self.new_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.terminal_memory[batch]

		return states, states_, actions, rewards, dones

