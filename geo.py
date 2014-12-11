import numpy as np
import matplotlib.pyplot as plt


def sign(p1, p2, p3):
	return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);

def in_tri(tri, p):
    v1 = tri[0]
    v2 = tri[1]
    v3 = tri[2]
    b1 = sign(p, v1, v2) < 0
    b2 = sign(p, v2, v3) < 0
    b3 = sign(p, v3, v1) < 0
    return b1 == b2 == b3

def in_core(core, p):
	for tri in core:
		if in_tri(tri, p):
			return True
	return False

def num_in_core(core1, inside):
	count = 0
	counter = 0
	for p in inside:
		print counter/float(len(core))
		counter += 1
		if in_core(core1, p):
			count += 1
	return count

def get_rect(tri):
	min_x = min(tri[:,0])
	max_x = max(tri[:,0])
	min_y = min(tri[:,1])
	max_y = max(tri[:,1])
	return np.array([[min_x,max_x],[min_y,max_y]])

def count_intersect(ref, core, tlx, tly, h, w):
	max_intersect = 0
	best_i = None
	inside = []
	# Get points inside core:
	for i in range(tly, tly + h):
		for j in range(tlx, tlx + w):
			print i,j
			if in_core(core, [i,j]):
				inside.append([i,j])
	for i, ref_core in enumerate(ref):
		num = num_in_core(core,inside)
		if num > max_intersect:
			best_i = i
	return max_intersect, best_i


