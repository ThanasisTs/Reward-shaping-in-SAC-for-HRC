import numpy as np
import time
from datetime import timedelta
from tqdm import tqdm
import pygame as pg
import math

class Experiment:
	def __init__(self, env, agent=None, trained_agent=None, load_models=False, config=None):
		self.env = env
		self.config = config
		self.agent = agent
		self.trained_agent = trained_agent
		self.best_score = None
		self.best_game_reward = None
		self.total_steps = 0
		self.action_history = []
		self.score_history = []
		self.game_duration_list = []
		self.grad_updates_durations = []
		self.test_score_history = []
		self.test_game_duration_list = []
		if load_models:
			self.agent.load_models()
		self.avg_grad_updates_duration = 0
		self.human_actions = [0, 0]
		self.agent_actions = [0, 0]
		self.save_models = None
		self.avg_length = 0
		self.update_cycles = self.config['Experiment']['total_update_cycles']
		self.action_duration = self.config['Experiment']['action_duration']
		self.max_episodes = self.config['Experiment']['max_episodes']

		# The max number of timesteps depends on the maximum episode duration. Each loop (human action, agent action,
		# environment update) needs approximately 16ms.
		self.max_timesteps = int(self.config['Experiment']['max_duration']/0.016)
		self.best_score = -100 -1 * self.max_timesteps
		self.best_game_reward = self.best_score
		self.randomness_threshold = self.config['Experiment']['stop_random_agent']
		self.test_agent_flag = False
		self.log_interval = self.config['Experiment']['log_interval']

		self.ppr_threshold = 0.7

		# The max number of timesteps depends on the maximum episode duration. Each loop (human action, agent action,
		# environment update) needs approximately 16ms.
		self.test_max_episodes = self.config['Experiment']['test']['max_episodes']
		self.test_max_timesteps = int(self.config['Experiment']['test']['max_duration']/0.016)
		self.test_best_score = -100 -1 * self.test_max_timesteps
		self.test_best_game_reward = self.test_best_score
		self.test_score_history = []
		self.test_game_duration_list = []

	def run(self):
		running_reward = 0 # total cumulative reward across all games
		current_timestep = 0 # current timestep for each game
		avg_length = 0
		for i_episode in range(1, self.max_episodes + 1):
			# At the beginning of each game, reset the environment and several variables
			self.env.first_time = True
			observation = self.env.get_state() # stores the state of the environment
			self.env.timeout = False # used to check if we hit the maximum game duration
			game_reward = 0 # keeps track of th rewards for each game
			current_timestep = 0 # timesteps for each episode

			print(f'Episode: {i_episode}')

			actions = [0, 0, 0, 0] # stores the pressed keys. [0, 0, 0, 0] means that no key is pressed
			self.save_models = True # flag for saving RL models
			tmp_time = 0 # used to check if 200ms have passed and the agent needs to take a new action

			for timestep in range(1, self.max_timesteps + 1):
				current_timestep += 1

				# compute agent's action every 200ms
				if time.time() - tmp_time > self.action_duration:
					tmp_time = time.time()
					randomness_criterion = i_episode
					self.compute_agent_action(observation, randomness_criterion)

				# get human action
				self.getKeyboard(actions)

				# get joint action
				action = self.get_action_pair()

				# check if maximum number of timesteps have ellapsed
				if timestep == self.max_timesteps:
					self.env.timeout = True

				# apply co-actions
				observation_, reward, done = self.env.step(action)

				if timestep == 1:
					start = time.time()

				# add experience to buffer
				interaction = [observation, self.agent_action, reward, observation_, done]
				self.save_experience(interaction)

				running_reward += reward
				game_reward += reward

				# online learning
				if not self.config['Game']['test_model']:
					if self.config['Experiment']['online_updates'] and i_episode > self.config['Experiment']['start_training_on_episode']:
						self.agent.learn()
						self.agent.soft_upadte_target()

				# update observed env state
				observation = observation_

				# if the game ended, proceed with the next game
				if done:
					break

			# keep track of the best game reward
			if self.best_game_reward < game_reward:
				self.best_game_reward = game_reward

			# keep track of the game reward history
			self.score_history.append(game_reward)

			# keep track of the game duration
			self.game_duration_list.append(time.time() - start)

			avg_length += current_timestep
			avg_ep_duration = np.mean(self.game_duration_list[-self.log_interval:])

			self.ppr_threshold -= 0.01

			# offline learning
			if not self.config['Game']['test_model'] and i_episode >= self.config['Experiment']['start_training_on_episode']:
				if i_episode % self.agent.update_interval == 0:
					self.updates_scheduler()
					if self.update_cycles > 0:
						grad_updates_duration = self.grad_updates(self.update_cycles)
						self.grad_updates_durations.append(grad_updates_duration)

						# save the models after each grad update
						self.agent.save_models()

					if i_episode % self.config['Experiment']['test_interval'] == 0 and self.test_max_episodes > 0:
						self.test_agent()

			if not self.config['Game']['test_model']:
				running_reward, avd_length = self.print_logs(i_episode, running_reward, avg_length, avg_ep_duration)


		if update_cycles > 0:
			try:
				self.avg_grad_updates_duration = np.mean(self.grad_updates_durations)
			except:
				print("Exception when calc grad_updates_durations")


	def test_agent(self):
		print('Testing agent')
		test_running_reward = 0 # total cumulative reward across all games
		avg_length = 0
		self.test_agent_flag = True

		for i_episode in range(1, self.test_max_episodes + 1):
			# At the beginning of each game, reset the environment and several variables
			self.env.first_time = True
			observation = self.env.get_state() # stores the state of the environment
			self.env.timeout = False # used to check if we hit the maximum game duration
			test_game_reward = 0 # keeps track of th rewards for each game

			actions = [0, 0, 0, 0] # stores the pressed keys. [0, 0, 0, 0] means that no key is pressed
			tmp_time = 0 # used to check if 200ms have passed and the agent needs to take a new action
			test_game_reward = 0 # total reward per episode

			for timestep in range(1, self.test_max_timesteps + 1):
				# compute agent's action every 200ms
				if time.time() - tmp_time > self.action_duration:
					tmp_time = time.time()
					randomness_criterion = i_episode
					self.compute_agent_action(observation, randomness_criterion)

				# get human action
				self.getKeyboard(actions)

				# get joint action
				action = self.get_action_pair()

				# check if maximum number of timesteps have ellapsed
				if timestep == self.test_max_timesteps:
					self.env.timeout = True

				# apply co-actions
				observation_, reward, done = self.env.step(action)

				if timestep == 1:
					start = time.time()

				test_running_reward += reward
				test_game_reward += reward

				# update observed env state
				observation = observation_

				# if the game ended, proceed with the next game
				if done:
					break

			# keep track of the best game reward
			if self.test_best_game_reward < test_game_reward:
				self.test_best_game_reward = test_game_reward

			# keep track of the game reward history
			self.test_score_history.append(test_game_reward)

			# keep track of the game duration
			self.test_game_duration_list.append(time.time() - start)
			self.test_agent_flag = False
			
	def getKeyboard(self, actions):
		pg.key.set_repeat(10)
		actions = [0, 0, 0, 0]
		space_pressed = True
		for event in pg.event.get():
			if event.type == pg.QUIT:
				return
			if event.type == pg.KEYDOWN:
				if event.key == pg.K_q:
					exit(1)
				if event.key in self.env.keys:
					actions = [0, 0, 0, 0]
					actions[self.env.keys[event.key]] = 1
			if event.type == pg.KEYUP:
				if event.key in self.env.keys:
					actions[self.env.keys[event.key]] = 0

		self.human_actions = actions[1] - actions[0]
	
	def get_action_pair(self):
		action = [self.agent_action, self.human_actions]
		self.action_history.append(action)
		return action

	def save_experience(self, interaction):
		self.agent.memory.add(*interaction)
	
	def grad_updates(self, update_cycles=None):
		start_grad_updates = time.time()
		end_grad_updates = 0
		print(f'Performing {update_cycles} updates')
		for _ in tqdm(range(update_cycles)):
			self.agent.learn()
			self.agent.soft_update_target()
			end_grad_updates = time.time()
		return end_grad_updates - start_grad_updates

	def print_logs(self, game, running_reward, avg_length, avg_ep_duration):
		if game % self.log_interval == 0:
			avg_length = int(avg_length / self.log_interval)
			log_reward = int((running_reward / self.log_interval))
			print(f'Episode {game}\tTotal timesteps {self.total_steps}\tavg length: {avg_length}\tTotal reward(last {self.log_interval} episodes): {log_reward}\tBest Score: {self.best_score}\tavg '
                'episode duration: {timedelta(seconds=avg_ep_duration)}')
			running_reward = 0
			avg_length = 0
		return running_reward, avg_length

	def compute_agent_action(self, observation, randomness_criterion=None):
		assert randomness_criterion != None, 'randomness_criterion is None'

		if self.test_agent_flag:
			self.agent_action = self.agent.actor.sample_act(observation)
		else:
			self.ppr_criterion = np.random.randint(100)/100
			if self.ppr_criterion < self.ppr_threshold:
				print('ppr')
				self.agent_action = self.trained_agent.actor.sample_act(observation)
				self.save_models = True
			else:
				print('e-greedy')
				if randomness_criterion <= self.randomness_threshold:
					# Pure exploration
					self.agent_action = np.random.randint(self.env.action_space.n_actions)
					self.save_models = False
				else:
					# Explore with actions_prob
					self.agent_action = self.agent.actor.sample_act(observation)
					self.save_models = True

	def updates_scheduler(self):
		total_update_cycles = self.config['Experiment']['total_update_cycles']
		online_updates = 0
		
		if self.config['Experiment']['online_updates']:
			online_updates = self.max_timesteps * (self.max_episodes - self.config['Experiment']['start_training_step_on_episode'])

		self.update_cycles = math.ceil((total_update_cycles - online_updates) / math.ceil(self.max_episodes / self.agent.update_interval))
