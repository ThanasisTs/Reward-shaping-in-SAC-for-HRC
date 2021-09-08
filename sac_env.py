# one direction if controlled by the human
# the other by a SAC agent
import os
import pygame as pg
import numpy as np
from scipy.spatial import distance
import random
import time
import matplotlib.pyplot as plt

from rewards import compute_reward

class ActionSpace:
	def __init__(self):
		self.actions = [i for i in range(3)]
		self.shape = 2
		self.n_actions = len(self.actions)

class Game:
	def __init__(self, controlled_variable="vel"):
		pg.init()

		self.display = (800, 800)
		self.screen = pg.display.set_mode(self.display)

		self.sleep_rate = 0.01

		self.start_pos = (100, 400)
		self.start_color = (0, 255, 0)

		self.end_pos = (400, 100)
		self.end_color = (255, 0, 0)

		self.radius = 30

		self.line_color = (0, 0, 255)

		self.min_pos, self.max_pos = 100, 700

		self.running = True
		self.keys = {pg.K_UP : 0, pg.K_DOWN : 1}
		self.random_actions = [-1, 0, 1]

		self.circle_color = (255, 165, 0)
		self.delay = 15
		self.win_dis = 10
		
		if controlled_variable == 'vel':
			self.cmd_vel = True
		else:
			self.cmd_vel = False
			self.min_vel = -2.0
			self.max_vel = 2.0

		self.images = {}
		image_scale = 0.2
		for img in os.listdir('images'):
			image = pg.image.load(os.path.join('images/' + img))

			image = pg.transform.scale(image, self.scaled_dim(image, image_scale))
			self.images.update({img : image})
		self.images = dict(sorted(self.images.items()))
		self.image_pos = (80, 10)

		self.timeout = False
		self.first_time = True

		self.start_game_time = time.time()
		self.max_duration = 10
		
		self.current_pos = np.array(self.start_pos, dtype=np.float64)
		self.current_vel = np.array([0, 0], dtype=np.float64)

		self.observation = self.get_state()
		self.action_space = ActionSpace()
		self.observation_space = (len(self.observation),)

	def get_state(self):
		return np.concatenate((self.current_pos, self.current_vel), axis=None)

	def scaled_dim(self, image, scale):
		img_size = image.get_size()
		return (int(np.ceil(scale*img_size[0])), int(np.ceil(scale*img_size[1])))

	def goal_dis(self):
		return distance.euclidean(self.current_pos, self.end_pos)

	def reset_game(self):
		self.goal_reached = False
		self.current_pos = np.array(self.start_pos, dtype=np.float64)
		self.current_vel = np.array([0, 0], dtype=np.float64)
		self.actions = [0, 0, 0, 0]
		self.trail = []
		self.done = False
		i = 0
		if not self.first_time:
			if self.timeout:
				self.screen.blit(list(self.images.values())[-1], self.image_pos)
				self.timeout = False
			else:
				self.screen.blit(list(self.images.values())[-3], self.image_pos)
			pg.display.flip()
			time.sleep(3)
		else:
			self.first_time = False
		self.start_time = time.time()
		while time.time() - self.start_time <= 5:
			self.screen.fill((105,105,105))
			pg.draw.circle(self.screen, self.start_color, self.start_pos, self.radius)
			pg.draw.circle(self.screen, self.end_color, self.end_pos, self.radius)
			pg.draw.line(self.screen, self.line_color, self.start_pos, self.end_pos)
			pg.draw.circle(self.screen, self.circle_color, self.current_pos, 5)
			self.screen.blit(list(self.images.values())[4-i], self.image_pos)
			i += 1
			pg.display.flip()
			time.sleep(1)
			
		self.screen.blit(self.images['play.png'], self.image_pos)
		pg.display.flip()
		time.sleep(1)
		pg.event.clear()
		self.start_game_time = time.time()

	def plot(self):
		fig = plt.figure()
		ax = plt.axes()
		ax.plot(*list(zip(*self.trail)), c='red')
		ax.grid()
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_xlim(self.min_pos, self.max_pos)
		ax.set_ylim(self.min_pos, self.max_pos)
		ax.invert_yaxis()
		plt.show()

	def step(self, action):
		if self.first_time:
			self.reset_game()
			
		time.sleep(self.sleep_rate)
		self.screen.fill((105,105,105))
		pg.draw.circle(self.screen, self.start_color, self.start_pos, self.radius)
		pg.draw.circle(self.screen, self.end_color, self.end_pos, self.radius)
		pg.draw.line(self.screen, self.line_color, self.start_pos, self.end_pos)
		pg.draw.circle(self.screen, self.circle_color, self.current_pos, 5)

		if self.cmd_vel:
			self.current_vel = np.array(action)
			self.current_vel[self.current_vel==2] = -1
		else:
			self.current_acc = np.array(action, dtype=np.float64)
			self.current_acc[self.current_acc == 2] = -1.0
			self.current_vel += 0.03*self.current_acc
			self.current_vel = np.clip(self.current_vel, self.min_vel, self.max_vel)

		self.current_pos += self.current_vel
		self.current_pos = np.clip(self.current_pos, self.min_pos, self.max_pos)
		self.trail.append(self.current_pos)
		pg.display.flip()

		if self.goal_dis() < self.win_dis or self.timeout:
			if self.goal_dis() < self.win_dis:
				self.goal_reached = True
			self.done = True

		reward = compute_reward(self.goal_reached)
		return self.get_state(), reward, self.done
