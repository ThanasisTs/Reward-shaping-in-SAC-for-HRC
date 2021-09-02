# one direction if controlled by the human
# the other by a random agent
import os
import pygame as pg
import numpy as np
from scipy.spatial import distance
import random
import time
import matplotlib.pyplot as plt

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
		self.keys_human = {pg.K_UP : 0, pg.K_DOWN : 1}
		self.random_actions = [-1, 0, 1]

		self.circle_color = (255, 165, 0)
		self.actions = [0, 0, 0, 0]
		self.delay = 15
		self.current_pos = list(self.start_pos)
		self.win_dis = 10

		self.images = {}
		for img in os.listdir('images'):
			image = pg.image.load(os.path.join('images/' + img))

			image = pg.transform.scale(image, self.scaled_dim(image, 0.3))
			self.images.update({img : image})
		self.images = dict(sorted(self.images.items()))

		self.timeout = False
		self.first_time = True

		self.reset_game()
		self.start_game_time = time.time()
		self.trail = []


	def scaled_dim(self, image, scale):
		img_size = image.get_size()
		return (int(np.ceil(scale*img_size[0])), int(np.ceil(scale*img_size[1])))

	def goal_dis(self):
		return distance.euclidean(self.current_pos, self.end_pos)

	def reset_game(self):
		self.current_pos = list(self.start_pos)
		self.actions = [0, 0, 0, 0]
		self.trail = []
		i = 0
		if not self.first_time:
			if self.timeout:
				self.screen.blit(list(self.images.values())[-1], (50, 150))
				self.timeout = False
			else:
				self.screen.blit(list(self.images.values())[-3], (50, 150))
			pg.display.flip()
			time.sleep(3)
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

	def getRandomAction(self):
		random_action = random.choice(self.random_actions)
		if random_action == 1:
			self.actions[2] = 1
		elif random_action == -1:
			self.actions[3] = 1
		else:
			self.actions[2:] = [0, 0]

	def getHumanAction(self):
		pg.key.set_repeat(self.delay)
		for event in pg.event.get():
			if event.type == pg.QUIT:
				self.running = False
			if event.type == pg.KEYDOWN:
				if event.key in self.keys_human:
					self.actions[self.keys_human[event.key]] = 1
			if event.type == pg.KEYUP:
				if event.key in self.keys_human:
					self.actions[self.keys_human[event.key]] = 0

	def run(self):
		while self.running:
			time.sleep(self.sleep_rate)
			self.screen.fill((105,105,105))
			pg.draw.circle(self.screen, self.start_color, self.start_pos, self.radius)
			pg.draw.circle(self.screen, self.end_color, self.end_pos, self.radius)
			pg.draw.line(self.screen, self.line_color, self.start_pos, self.end_pos)
			pg.draw.circle(self.screen, self.circle_color, self.current_pos, 5)

			self.getRandomAction()
			self.getHumanAction()

			self.current_pos[1] += self.actions[1] - self.actions[0]
			self.current_pos[0] += self.actions[3] - self.actions[2]
			self.current_pos = np.clip(self.current_pos, self.min_pos, self.max_pos)
			self.trail.append(self.current_pos)

			if time.time() - self.start_game_time >= 2:
				self.timeout = True

			if self.goal_dis() < self.win_dis or self.timeout:
				self.plot()
				self.reset_game()

			pg.display.flip()


if __name__ == '__main__':
	game = Game()
	game.run()