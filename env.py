import os
import pygame as pg
import numpy as np
from scipy.spatial import distance
import time

def scaled_dim(image, scale):
	return (int(np.ceil(scale*image.get_size()[0])), int(np.ceil(scale*image.get_size()[1])))

def goal_dis(x, y):
	return distance.euclidean((x, y), end_pos)

def reset_game(first_time = False):
	global current_x, current_y, keys_pressed, event, timeout
	current_x, current_y = start_pos[0], start_pos[1]
	keys_pressed = [0, 0, 0, 0]
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
		pg.draw.circle(screen, circle_color, (current_x, current_y), 5)
		screen.blit(list(images.values())[4-i], (50, 150))
		i += 1
		pg.display.flip()
		time.sleep(1)
		
	screen.blit(images['play.png'], (50, 150))
	pg.display.flip()
	time.sleep(1)
	pg.event.clear()

pg.init()

display = (800, 800)
screen = pg.display.set_mode(display)

start_pos = (100, 700)
start_color = (0, 255, 0)

end_pos = (700, 100)
end_color = (255, 0, 0)

radius = 30

line_color = (0, 0, 255)

update_pos = 2

run = True
keys = {pg.K_UP : 0, pg.K_DOWN : 1, pg.K_LEFT : 2, pg.K_RIGHT : 3}

circle_color = (255, 165, 0)
keys_pressed = [0, 0, 0, 0]
delay = 15
current_x, current_y = start_pos[0], start_pos[1]

images = {}
for img in os.listdir('images'):
	image = pg.image.load(os.path.join('images/' + img))

	image = pg.transform.scale(image, scaled_dim(image, 0.3))
	images.update({img : image})
images = dict(sorted(images.items()))

timeout = False

reset_game(first_time = True)
start_game_time = time.time()

while run:
	pg.key.set_repeat(delay)


	screen.fill((105,105,105))
	pg.draw.circle(screen, start_color, start_pos, radius)
	pg.draw.circle(screen, end_color, end_pos, radius)
	pg.draw.line(screen, line_color, start_pos, end_pos)
	pg.draw.circle(screen, circle_color, (current_x, current_y), 5)

	for event in pg.event.get():
		if event.type == pg.QUIT:
			run = False
		if event.type == pg.KEYDOWN:
			if event.key in keys:
				keys_pressed[keys[event.key]] = 1
		if event.type == pg.KEYUP:
			if event.key in keys:
				keys_pressed[keys[event.key]] = 0

		current_y -= keys_pressed[0]
		current_y += keys_pressed[1]

		current_x -= keys_pressed[2]
		current_x += keys_pressed[3]

	if time.time() - start_game_time >= 10:
		timeout = True

	if goal_dis(current_x, current_y) < 10 or timeout:
		reset_game()
		start_game_time = time.time()

	pg.display.flip()
