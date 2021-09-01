import os
import pygame as pg
import numpy as np
from scipy.spatial import distance
import random
import time
import matplotlib.pyplot as plt

def scaled_dim(image, scale):
	img_size = image.get_size()
	return (int(np.ceil(scale*img_size[0])), int(np.ceil(scale*img_size[1])))

def goal_dis(pos):
	return distance.euclidean(pos, end_pos)

def reset_game(first_time = False):
	global current_pos, actions, event, timeout, trail
	current_pos = list(start_pos)
	actions = [0, 0, 0, 0]
	trail = []
	i = 0
	if not first_time:
		if timeout:
			screen.blit(list(images.values())[-1], (50, 150))
			timeout = False
		else:
			screen.blit(list(images.values())[-3], (50, 150))
		pg.display.flip()
		time.sleep(3)
	start_time = time.time()
	while time.time()-start_time <= 5:
		screen.fill((105,105,105))
		pg.draw.circle(screen, start_color, start_pos, radius)
		pg.draw.circle(screen, end_color, end_pos, radius)
		pg.draw.line(screen, line_color, start_pos, end_pos)
		pg.draw.circle(screen, circle_color, current_pos, 5)
		screen.blit(list(images.values())[4-i], (50, 150))
		i += 1
		pg.display.flip()
		time.sleep(1)
		
	screen.blit(images['play.png'], (50, 150))
	pg.display.flip()
	time.sleep(1)
	pg.event.clear()

def plot(trail):
	fig = plt.figure()
	ax = plt.axes()
	ax.plot(*list(zip(*trail)), c='red')
	ax.grid()
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_xlim(min_pos, max_pos)
	ax.set_ylim(min_pos, max_pos)
	ax.invert_yaxis()
	plt.show()

def getRandomAction():
	global actions, random_actions
	random_action = random.choice(random_actions)
	if random_action == 1:
		actions[2] = 1
	elif random_action == -1:
		actions[3] = 1
	else:
		actions[2:] = [0, 0]

def getHumanAction(delay=10):
	global keys_human, actions
	pg.key.set_repeat(delay)
	for event in pg.event.get():
		if event.type == pg.QUIT:
			run = False
		if event.type == pg.KEYDOWN:
			if event.key in keys_human:
				actions[keys_human[event.key]] = 1
		if event.type == pg.KEYUP:
			if event.key in keys_human:
				actions[keys_human[event.key]] = 0

pg.init()

display = (800, 800)
screen = pg.display.set_mode(display)

sleep_rate = 0.01

start_pos = (100, 700)
start_color = (0, 255, 0)

end_pos = (700, 100)
end_color = (255, 0, 0)

radius = 30

line_color = (0, 0, 255)

min_pos, max_pos = 100, 700

run = True
keys_human = {pg.K_UP : 0, pg.K_DOWN : 1}
random_actions = [-1, 0, 1]

circle_color = (255, 165, 0)
actions = [0, 0, 0, 0]
delay = 15
current_pos = list(start_pos)

images = {}
for img in os.listdir('images'):
	image = pg.image.load(os.path.join('images/' + img))

	image = pg.transform.scale(image, scaled_dim(image, 0.3))
	images.update({img : image})
images = dict(sorted(images.items()))

timeout = False

reset_game(first_time = True)
start_game_time = time.time()
trail = []

while run:
	time.sleep(sleep_rate)
	screen.fill((105,105,105))
	pg.draw.circle(screen, start_color, start_pos, radius)
	pg.draw.circle(screen, end_color, end_pos, radius)
	pg.draw.line(screen, line_color, start_pos, end_pos)
	pg.draw.circle(screen, circle_color, current_pos, 5)

	getRandomAction()
	getHumanAction(delay)

	current_pos[1] += actions[1] - actions[0]
	current_pos[0] += actions[3] - actions[2]
	current_pos = np.clip(current_pos, min_pos, max_pos)
	trail.append(current_pos)

	if time.time() - start_game_time >= 2:
		timeout = True

	if goal_dis(current_pos) < 10 or timeout:
		plot(trail)
		reset_game()
		start_game_time = time.time()

	pg.display.flip()
