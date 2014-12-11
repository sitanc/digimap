import Image
import random
import numpy as np
import main
import json
from sklearn import svm
from params import *
import matplotlib.pyplot as plt
import hog
from scipy.ndimage import filters
import copy
import pickle

# Open map image
im = Image.open('map_backup.jpg').convert('1')
x = im.load()
w,h = im.size

# Load classifier if exists, otherwise train
# NOTE: I removed the training/ folder from github
# repo because source of data prohibited
# public dissemination
try:
	with open('clf.pkl', 'rb') as f:
		clf = pickle.load(f)
	print "CLF opened!"
except:
	labs_file = open('training/all_labs', 'rb')
	feat_file = open('training/all_hogs', 'rb')
	print 'Data loaded!'
	features = np.array(json.load(feat_file))
	s = random.sample(range(len(features)), 30000)
	features = features[s,:]
	labels = np.array(json.load(labs_file))[s]
	clf = svm.SVC(gamma=0.01, C = 100.)
	clf.fit(features, labels)
	with open('clf.pkl', 'wb') as f:
		pickle.dump(clf, f)
print 'SVM done!'

# Returns matrix corresponding to window starting at
# (corner_x,corner_y) with dimensions W x W
def get_window(x,corner_x, corner_y):
	window = np.zeros((W,W))
	points = []
	for i in range(corner_x, corner_x+W):
		for j in range(corner_y, corner_y+W):
			window[j-corner_y, i-corner_x] = x[i,j]
	return window

# Smooths a window using a Gaussian filter
def smooth_window(window, sigma=2):
	return filters.gaussian_filter(window, sigma)

# Retrieves a list of (slightly randomly perturbed) points
# corresponding to black pixels in window, starting at
# (corner_x,corner_y), and of dimensions W x W
# Optional: if border = True, add a cloud of border points to
# the window
def get_points(window, corner_x, corner_y, W, border=True):
	pts = []
	if border:
		for i in range(1,W-1):
			perturbs = np.random.random(4)
			if random.random() < 0.25:
				pts.append(np.array([i+corner_x, corner_y + perturbs[0]]))
			if random.random() < 0.25:
				pts.append(np.array([i+corner_x, corner_y + W-1 + perturbs[1]]))
			if random.random() < 0.25:
				pts.append(np.array([corner_x+ perturbs[2], corner_y + i ]))
			if random.random() < 0.25:
				pts.append(np.array([corner_x + W-1+ perturbs[3], corner_y + i]))
	for x in range(corner_x, corner_x+W):
		for y in range(corner_y, corner_y+W):
			if window[y-corner_y,x-corner_x] < 1:
				pert_x = 2*random.random()-1
				pert_y = 2*random.random()-1
				pts.append(np.array([x+pert_x,y+pert_y]))
	pts = np.array(pts)
	plt.scatter(pts[:,0], -pts[:,1],s=0.000001); plt.show()
	return pts

# Detects and removes rid of text in image. Algorithm based on:
# http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6637990
def erase(window, offset_x, offset_y, efficient=None):
	# Retrieve HoG features, gradient magnitudes, and mask new_labs
	# indicating whether each pixel is a text pixel
	if efficient:
		try:
			with open('memos/test_hogs' + efficient + '.pkl', 'rb') as f:
				hogs = pickle.load(f)
		except:
			hogs = hog.get_hog(window)
			with open('memos/test_hogs' + efficient + '.pkl', 'wb') as f:
				pickle.dump(hogs, f)
		try:
			with open('memos/test_new_labs' + efficient + '.pkl', 'rb') as f:
				new_labs = pickle.load(f)
		except:
			new_labs = clf.predict(hogs.reshape(W**2, n_bins))
			new_labs = new_labs.reshape(W,W)
			with open('memos/test_new_labs' + efficient + '.pkl', 'wb') as f:
				pickle.dump(new_labs, f)
		try:
			with open('memos/test_mags' + efficient + '.pkl', 'rb') as f:
				mags = pickle.load(f)
		except:
			mags = hog.get_magnitudes(window)
			with open('memos/test_mags' + efficient + '.pkl', 'wb') as f:
				pickle.dump(mags, f)
	else:
		hogs = hog.get_hog(window)
		new_labs = clf.predict(hogs.reshape(W**2, n_bins))
		new_labs = new_labs.reshape(W,W)
		mags = hog.get_magnitudes(window)

	# Ignore positive labels if gradient magnitude is zero
	new_labs[mags == 0] = 0

	# Remove from window all pixels where all of the following are satisfied:
	# 1) support of HoG is more than n (=8) or the SVM says it's text
	# 2) gradient magnitude is sufficiently large
	# 3) pixel value in window is not white
	window_new = copy.copy(window)
	text = np.ones((W,W))
	for i in range(W):
		for j in range(W):
			if ((sum(hogs[i,j,:] > 0) > n or new_labs[i,j] == 1) and
				mags[i,j] > mag_thres and window[i,j] < 255):
				window_new[i,j] = 255
				text[i,j] = 0
	# Smooth the result and get rid of grayish pixels
	smoothed = smooth_window(window_new, sigma);
	smoothed[smoothed > 1] = 255
	# Obtain point cloud
	smoothed_points = get_points(smoothed,offset_x,offset_y,W, border=False);
	# Plot, if you want
	plt.imshow(smoothed); plt.show()
	return smoothed, smoothed_points, new_labs

# Outputs matrices I given by I(i,j) = number of pixels with
# color i in color_map1 and j in color_map2 in overlap, and vice versa
# d is the directional relationship between the adjacent matrices
# NOTE: color_map1 is on top of or to the left of color_map2
def count_intersect(color_map1, color_map2, d):
	# Returns list of unique values of entries in a matrix
	def values(patch):
		return list(set(patch.reshape(1,W*W).tolist()[0]))
	h,w = np.shape(color_map1)
	values1 = values(color_map1)
	values2 = values(color_map2)
	intersects12 = {value:{} for value in values1}
	intersects21 = {value:{} for value in values2}
	# 1 is to the left of 2
	if d == "h":
		color_map1 = color_map1[:,int(w*overlap):]
		color_map2 = color_map2[:,:int(w*(1-overlap))]
	# 1 is above 2
	else:
		color_map1 = color_map1[int(h*overlap):,:]
		color_map2 = color_map2[:int(h*(1-overlap)),:]
	# for each pair of colors, check how many points in intersection
	# have those two colors in color_map1 and color_map2 respectively
	for i in values1:
		for j in values2:
			intersects12[i][j] = np.sum((color_map1 == i) & (color_map2 == j))
			intersects21[j][i] = np.sum((color_map1 == i) & (color_map2 == j))
	return intersects12, intersects21

# Local-to-global reconstruction algorithm
# Takes collection of segmentation matrices for individual window
# and outputs single segmentation matrix for entire image
def loc_to_glob(all_colors):
	# Pastes patch at row r and column c into final matrix
	def paste_in(patch, r, c, thres,overlap):
		h,w = np.shape(patch)
		I = int((r - rs[0])/(W*overlap))
		J = int((c - cs[0])/(W*overlap))
		values2 = values(patch)
		rgx = (0,int((1+overlap)/2.*W))
		rgy = (0,int((1+overlap)/2.*W))
		new_patch = copy.copy(patch)
		if I > 0:
			top = all_colors[J,I-1]
			values_top = values(top)
			intersects_top12, intersects_top21 = count_intersect(top, patch, 'v')
			for i in values_top:
				for j in values2:
					rat1 = intersects_top12[i][j] / float(sum(intersects_top12[i].values()))
					rat2 = intersects_top21[j][i] / float(sum(intersects_top21[j].values()))
					print rat1, rat2
					if rat1 > thres and rat2 > thres:
						new_patch[patch == j] = i
			if I == len(rs) - 1:
				rgy = (int((1-overlap)/2*W),W)
			else:
				rgy = (int((1-overlap)/2*W),int((1+overlap)/2*W))
		if J > 0:
			left = all_colors[J-1,I]
			values_left = values(left)
			intersects_left12, intersects_left21 = count_intersect(left, patch, 'h')
			for i in values_left:
				for j in values2:
					rat1 = intersects_left12[i][j] / float(sum(intersects_left12[i].values()))
					rat2 = intersects_left21[j][i] / float(sum(intersects_left21[j].values()))
					if rat1 > thres and rat2 > thres:
						new_patch[patch == j] = i
			if J == len(cs) - 1:
				rgx = (int((1-overlap)/2*W),W)
			else:
				rgx = (int((1-overlap)/2*W),int((1+overlap)/2*W))
		canvas[r-rs[0]+rgy[0]:r-rs[0]+rgy[1],c-cs[0]+rgx[0]:c-cs[0]+rgx[1]] = new_patch[rgy[0]:rgy[1],rgx[0]:rgx[1]]
		# plt.imshow(canvas); plt.show()
		all_colors[J,I] = new_patch
	canvas = np.zeros((2*W, 2*W))
	for i in range(len(cs)):
		for j in range(len(rs)):
			patch = all_colors[i,j]
			r = rs[0] + j*500*overlap
			c = cs[0] + i*500*overlap
			paste_in(patch, r, c, 0.7, overlap)
	plt.imshow(canvas); plt.show()
	return canvas

# Input: coordinates, output: segmentation
def process(cs,rs):
	# master is the collection of segmentation matrices for all windows
	master = np.zeros((len(cs),len(rs),500,500))
	for i, c in enumerate(cs):
		for j, r in enumerate(rs):
			window = get_window(x, c, r)
			new_window, points, is_text = erase(window,c,r,efficient=str(i) + str(j))
			m, cores, simplex_map, tri = main.main(points, None)
			seg_map = main.window_seg(W,W,c,r,tri,simplex_map)
			master[i,j] = seg_map
	master_copy = copy.copy(master)
	palette = 1000*np.random.random((len(cs),len(rs),np.max(master_copy)+2))
	all_colors = np.zeros(np.shape(master_copy))
	return loc_to_glob(all_colors)

# # Coordinates corresponding to map.jpg
# cs = range(2800,3301,125)
# rs = range(3100,3601,125)
# process(cs,rs)