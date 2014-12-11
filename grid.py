# grid.py
# Code for testing our segmentation algorithm
# Generates noisy square grids

from random import *
import matplotlib.pyplot as plt
import numpy as np

# Samples 'size' points from w x w grid
# with noise of size 'radius'
def grid(w, radius, size):
	points = []
	xs = []
	ys = []
	for _ in range(size):
		row = randint(1,2)
		number = randint(0,w)
		location = random()*(w)
		if row == 1:
			p = [location, number]
		else:
			p = [number, location]
		offsetx = (2*random()-2)*radius
		offsety = (2*random()-2)*radius
		q1 = p[0] + offsetx
		q2 = p[1] + offsety
		points.append((q1,q2))
		xs.append(q1)
		ys.append(q2)
	plt.scatter(xs,ys)
	plt.show()
	return np.array(points)

# Creates grid of points sampled along lattice points
# as in pixels in an image
# NOTE: used this to show that segmentation doesn't work if
# we don't add some random noise to points first
def grid_int(h,w,scale):
	points = []
	xs = []
	ys = []
	for i in range(h):
		for j in range(w):
			if i == 0 or j == 0 or i == h-1 or j == w - 1:
				x = i + random()/scale
				y = j + random()/scale
				points.append((x,y))
				xs.append(x)
				ys.append(y)
	plt.scatter(xs,ys)
	plt.show()
	return np.array(points)
