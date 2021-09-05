# one direction if controlled by the human
# the other by a SAC agent
import os
import pygame as pg
import numpy as np
from scipy.spatial import distance
import random
import time
import matplotlib.pyplot as plt

class ActionSpace:
	def __init__(self):
		self.actions = [i for i in range(3)]
		self.shape = 2
		self.n_actions = len(self.actions)

class Game:
	def __init__(self):
		pg.init()

		self.display = (800, 800)
		self.screen = pg.display.set_mode(self.display)

		self.sleep_rate = 0.01

		self.start_pos = (100, 700)
		self.start_color = (0, 255, 0)

		self.end_pos = (700, 100)
		self.end_color = (255, 0, 0)

		self.radius = 30

		self.line_color = (0, 0, 255)

		self.min_pos, self.max_pos = 100, 700

		self.running = True
		self.keys = {pg.K_UP : 0, pg.K_DOWN : 1}
		self.random_actions = [-1, 0, 1]

		self.circle_color = (255, 165, 0)
		self.actions = [0, 0, 0, 0]
		self.delay = 15
		self.current_pos = list(self.start_pos)
		self.current_vel = [0, 0]
		self.win_dis = 10

		self.images = {}
		for img in os.listdir('images'):
			image = pg.image.load(os.path.join('images/' + img))

			image = pg.transform.scale(image, self.scaled_dim(image, 0.3))
			self.images.update({img : image})
		self.images = dict(sorted(self.images.items()))

		self.timeout = False
		self.first_time = True

		self.start_game_time = time.time()
		self.trail = []
		self.max_duration = 10

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
		self.current_pos = list(self.start_pos)
		self.actions = [0, 0, 0, 0]
		self.trail = []
		self.done = False
		i = 0
		if not self.first_time:
			if self.timeout:
				self.screen.blit(list(self.images.values())[-1], (50, 150))
				self.timeout = False
			else:
				self.screen.blit(list(self.images.values())[-3], (50, 150))
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
			self.screen.blit(list(self.images.values())[4-i], (50, 150))
			i += 1
			pg.display.flip()
			time.sleep(1)
			
		self.screen.blit(self.images['play.png'], (50, 150))
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
			
		self.current_vel = np.array(action)
		self.current_vel[self.current_vel==2] = -1
		

		time.sleep(self.sleep_rate)
		self.screen.fill((105,105,105))
		pg.draw.circle(self.screen, self.start_color, self.start_pos, self.radius)
		pg.draw.circle(self.screen, self.end_color, self.end_pos, self.radius)
		pg.draw.line(self.screen, self.line_color, self.start_pos, self.end_pos)
		pg.draw.circle(self.screen, self.circle_color, self.current_pos, 5)

		self.current_pos[0] += self.current_vel[0]
		self.current_pos[1] += self.current_vel[1]
		self.current_pos = np.clip(self.current_pos, self.min_pos, self.max_pos)
		self.trail.append(self.current_pos)

		if self.goal_dis() < self.win_dis or self.timeout:
			self.done = True

		pg.display.flip()
		reward = 1
		return self.get_state(), reward, self.done


# if __name__ == '__main__':
# 	game = Game()
# 	game.step()