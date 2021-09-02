import numpy as np
import time
from datetime import timedelta
from tqdm import tqdm
import pygame as pg
import math

class Experiment:
	def __init__(self, env, agent=None, load_models=False, config=None):
		self.env = env
		self.config = config
		self.agent = agent
		self.best_score = None
		self.best_game_reward = None
		self.best_score_episode = -1
		self.best_score_length = -1
		self.total_steps = 0
		self.action_history = []
		self.score_history = []
		self.game_duration_list = []
		self.length_list = []
		self.grad_updates_durations = []
		self.test_length_list = []
		self.test_score_history = []
		self.test_game_duration_list = []
		if load_models:
			self.agent.load_models()
		self.max_episodes = None
		self.max_timesteps = None
		self.avg_grad_updates_duration = 0
		self.human_actions = [0, 0]
		self.agent_actions = [0, 0]
		self.total_timesteps = None
		self.max_timesteps_per_game = None
		self.save_models = None
		self.game = None
		self.update_cycles = None
		self.distane_travel_list = []
		self.reward_list = []
		self.test_reward_list = []
		self.last_time = 0
		self.key_pressed_count = 0
		self.last_pressed = None


	def run(self):
		flag = True
		current_timestep = 0 # current timestep for each game
		running_reward = 0 # total cumulative reward across all games
		avg_length = 0

		self.action_duration = self.config['Experiment']['action_duration']

		self.max_episodes = self.config['Experiment']['max_episodes']

		# The max number of timesteps depends on the maximum episode duration. Each loop (human action, agent action,
		# environment update) needs approximately 16ms.
		self.max_timesteps = int(self.config['Experiment']['max_duration']/0.016)
		self.best_score = -100 -1 * self.max_timesteps
		self.best_game_reward = self.best_score

		for i_episode in range(1, self.max_episodes + 1):
			# At the beginning of each game, reset the environment and several variables
			self.env.first_time = True
			observation = self.env.get_state() # stores the state of the environment
			reset = True # used to reset the graphics environment when a new game starts
			self.env.timeout = False # used to check if we hit the maximum game duration
			game_reward = 0 # keeps track of th rewards for each game
			dist_travel = 0 # keeps track of the ball's travelled distance
			test_offline_score = 0

			print(f'Episode: {i_episode}')

			actions = [0, 0, 0, 0] # stores the pressed keys. [0, 0, 0, 0] means that no key is pressed
			duration_pause = 0 # keeps track of the pause time
			self.save_models = True # flag for saving RL models
			tmp_time = 0 # used to check if 200ms have passed and the agent needs to take a new action

			for timestep in range(1, self.max_timesteps + 1):
				self.total_steps += 1
				current_timestep += 1

				# compute agent's action every 200ms
				if time.time() - tmp_time > self.action_duration:
					tmp_time = time.time()
					randomness_threshold = self.config['Experiment']['stop_random_agent']
					randomness_criterion = i_episode
					flag = self.compute_agent_action(observation, randomness_criterion, randomness_threshold, flag)

				# get human action
				self.getKeyboard(actions)

				action = self.get_action_pair()

				if timestep == self.max_timesteps:
					self.env.timeout = True

				# apply co-actions
				observation_, reward, done = self.env.step(action)

				if timestep == 1:
					start = time.time()

				if reset:
					reset = False

				# add experience to buffer
				interaction = [observation, self.agent_action, reward, observation_, done]
				self.save_experience(interaction)

				running_reward += reward
				game_reward += reward

				# online learning
				if not self.config['game']['test_model']:
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
			self.reward_list.append(game_reward)

			# keep track of the game duration
			self.game_duration_list.append(time.time() - start)

			# offline learning
			if not self.config['game']['test_model'] and i_episode >= self.config['Experiment']['start_training_on_episode']:
				if i_episode % self.agent.update_interval == 0:
					self.updates_scheduler()
					if self.update_cycles > 0:
						grad_updates_duration = self.grad_updates(self.update_cycles)
						self.grad_updates_durations.append(grad_updates_duration)

						# save the models after each grad update
						self.agent.save_models()

					if i_episode % self.config['Experiment']['test_interval'] == 0 and self.test_max_episodes > 0:
						self.test_agent()

					if not self.config['game']['test_model']:
						running_reward, avd_length = self.print_logs(i_episode, running_reward, avg_length, log_interval, avg_ep_duration)

					current_timestep = 0
		update_cycles = np.ceil(self.config['Experiment']['total_update_cycles'])
		if update_cycles > 0:
			try:
				self.avg_grad_updates_duration = np.mean(self.grad_updates_durations)
			except:
				print("Exception when calc grad_updates_durations")

	def getKeyboard(self, actions):
		pg.key.set_repeat(10)
		actions = [0, 0, 0, 0]
		space_pressed = True
		for event in pg.event.get():
			if event.type == pg.QUIT:
				return
			if event.type == pg.KEYDOWN:
				self.last_time = time.time()
				if event.key == pg.K_q:
					exit(1)
				if event.key in self.env.keys:
					self.key_pressed_count = 0
					self.last_pressed = self.env.keys[event.key]
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

	def print_logs(self, game, running_reward, avg_length, log_interval, avg_ep_duration):
		if game % log_interval == 0:
			avg_length = int(avg_length / log_interval)
			log_reward = int((running_reward / log_interval))
			print(f'Episode {game}\tTotal timesteps {self.total_steps}\tavg length: {avg_length}\tTotal reward(last {log_interval} episodes): {log_reward}\tBest Score: {self.best_score}\tavg '
                'episode duration: {timedelta(seconds=avg_ep_duration)}')
			running_reward = 0
			avg_length = 0
		return running_reward, avg_length

	def compute_agent_action(self, observation, randomness_critirion=None, randomness_threshold=None, flag=True):
		if randomness_critirion is not None and randomness_threshold is not None and randomness_critirion <= randomness_threshold:
			# Pure exploration
			if self.config['game']['agent_only']:
				self.agent_action = np.random.randint(pow(2, self.env.action_space.n_actions))
			else:
				self.agent_action = np.random.randint(self.env.action_space.n_actions)
			self.save_models = False
			if flag:
				print("Using Random Agent")
				flag = False
		else:
			# Explore with actions_prob
			self.save_models = True
			self.agent_action = self.agent.actor.sample_act(observation)
			if not flag:
				print("Using SAC Agent")
				flag = True
		if self.agent_action == 2:
			self.agent_action = -1

		return flag

	def get_agent_only_action(self):
		# up: 0, down:1, left:2, right:3, upleft:4, upright:5, downleft: 6, downright:7
		if self.agent_action == 0:
			return [1, 0]
		elif self.agent_action == 1:
			return [-1, 0]
		elif self.agent_action == 2:
			return [0, -1]
		elif self.agent_action == 3:
			return [0, 1]
		elif self.agent_action == 4:
			return [1, -1]
		elif self.agent_action == 5:
			return [1, 1]
		elif self.agent_action == 6:
			return [-1, -1]
		elif self.agent_action == 7:
			return [-1, 1]
		else:
			print("Invalid agent action")

	def updates_scheduler(self):
		update_list = [22000, 1000, 1000, 1000, 1000, 1000, 1000]
		total_update_cycles = self.config['Experiment']['total_update_cycles']
		online_updates = 0
		if self.config['Experiment']['online_updates']:
			online_updates = self.max_timesteps * (
				self.max_episodes - self.config['Experiment']['max_episodes_mode']['start_training_step_on_episode'])
		if self.update_cycles is None:
			self.update_cycles = total_update_cycles - online_updates
		if self.config['Experiment']['scheduling'] == "descending":
			self.counter += 1
			if not (math.ceil(self.max_episodes / self.agent.update_interval) == self.counter):
				self.update_cycles /= 2
		elif self.config['Experiment']['scheduling'] == "big_first":
			if self.config['Experiment']['online_updates']:
				if self.counter == 1:
					self.update_cycles = update_list[self.counter]
				else:
					self.update_cycles = 0
			else:
				self.update_cycles = update_list[self.counter]
				self.counter += 1
		else:
			print(self.max_episodes, self.agent.update_interval)
			self.update_cycles = (total_update_cycles - online_updates) / math.ceil(
				self.max_episodes / self.agent.update_interval)
		self.update_cycles = math.ceil(self.update_cycles)

	def convert_actions(self, actions):
	    # gets a list of 4 elements. it is called from getKeyboard()
	    action = []
	    if actions[0] == 1:
	        action.append(1)
	    elif actions[1] == 1:
	        action.append(2)
	    else:
	        action.append(0)
	    if actions[2] == 1:
	        action.append(1)
	    elif actions[3] == 1:
	        action.append(2)
	    else:
	        action.append(0)
	    return action