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

W = 500
mag_thres = 120
# Histogram must have at least 8 nonempty buckets to be text
n = 8
sigma = 0.6
overlap = 0.25

im = Image.open('map_backup.jpg').convert('1')
x = im.load()
w,h = im.size

try:
	with open('memos/window.pkl', 'rb') as f:
		window = pickle.load(f)
	print "OPENED WINDOW PICKLE"
except:
	im = Image.open('map.jpg').convert('1')
	print 'Image opened!'
	x = im.load()
	w,h = im.size
	corners_x = range(w,W)
	corners_y = range(h,W)
	with open('memos/window.pkl', 'wb') as f:
		pickle.dump(window, f)
	print "MADE NEW WINDOW"

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

def get_window(x,corner_x, corner_y):
	window = np.zeros((W,W))
	points = []
	for i in range(corner_x, corner_x+W):
		for j in range(corner_y, corner_y+W):
			window[j-corner_y, i-corner_x] = x[i,j]
	return window

def smooth_window(window, sigma=2):
	return filters.gaussian_filter(window, sigma)

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

# Gets rid of text
def erase(window, offset_x, offset_y, efficient=None):
	if efficient:
		try:
			with open('memos/test_hogs' + efficient + '.pkl', 'rb') as f:
				hogs = pickle.load(f)
			print "OPENED HOG PICKLE"
		except:
			hogs = hog.get_hog(window)
			with open('memos/test_hogs' + efficient + '.pkl', 'wb') as f:
				pickle.dump(hogs, f)

		try:
			with open('memos/test_new_labs' + efficient + '.pkl', 'rb') as f:
				new_labs = pickle.load(f)
			print "OPENED NEW_LABS PICKLE"
		except:
			new_labs = clf.predict(hogs.reshape(W**2, n_bins))
			new_labs = new_labs.reshape(W,W)
			with open('memos/test_new_labs' + efficient + '.pkl', 'wb') as f:
				pickle.dump(new_labs, f)
		try:
			with open('memos/test_mags' + efficient + '.pkl', 'rb') as f:
				mags = pickle.load(f)
			print "OPENED MAGS PICKLE"
		except:
			mags = hog.get_magnitudes(window)
			with open('memos/test_mags' + efficient + '.pkl', 'wb') as f:
				pickle.dump(mags, f)
	else:
		hogs = hog.get_hog(window)
		new_labs = clf.predict(hogs.reshape(W**2, n_bins))
		new_labs = new_labs.reshape(W,W)
		mags = hog.get_magnitudes(window)

	new_labs[mags == 0] = 0
	window_new = copy.copy(window)
	text = np.ones((W,W))
	for i in range(W):
		for j in range(W):
			if ((sum(hogs[i,j,:] > 0) > n or new_labs[i,j] == 1) and
				mags[i,j] > 0 and window[i,j] < 255):
				window_new[i,j] = 255
				text[i,j] = 0

	smoothed = smooth_window(window_new, sigma);
	# smoothed_points = get_points(smoothed,offset_x,offset_y,W, border=False);
	smoothed[smoothed > 1] = 255
	plt.imshow(smoothed); plt.show()
	return smoothed, smoothed_points, new_labs

# cs = range(2800,3301,125)
# rs = range(3100,3601,125)
cs = range(4350,4851,125)
rs = range(3150,3651,125)


def values(patch):
	return list(set(patch.reshape(1,W*W).tolist()[0]))

# 1 is on top of or to the left of 2
def count_intersect(color_map1, color_map2, d):
	h,w = np.shape(color_map1)
	values1 = values(color_map1)
	values2 = values(color_map2)
	intersects12 = {value:{} for value in values1}
	intersects21 = {value:{} for value in values2}
	if d == "h":
		color_map1 = color_map1[:,w/4:]
		color_map2 = color_map2[:,:3*w/4]
	elif d == "v":
		color_map1 = color_map1[h/4:,:]
		color_map2 = color_map2[:3*h/4,:]
	for i in values1:
		for j in values2:
			intersects12[i][j] = np.sum((color_map1 == i) & (color_map2 == j))
			intersects21[j][i] = np.sum((color_map1 == i) & (color_map2 == j))
	return intersects12, intersects21

canvas = np.zeros((2*W, 2*W))

def paste_in(patch, r, c, thres,overlap):
	h,w = np.shape(patch)
	I = int((r - rs[0])/(125.))
	J = int((c - cs[0])/(125.))
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

# for corner_x in corners_x:
# 	for corner_y in corners_y:
# 		window = get_window(corner_x, corner_y)
# 		is_text = process(window)
# 		points = get_points(is_text, corner_x, corner_y)
# 		cores = main.main(points)

master = np.zeros((len(cs),len(rs),500,500))

for i, c in enumerate(cs):
	for j, r in enumerate(rs):
		window = get_window(x, c, r)
		new_window, points, is_text = erase(window,c,r,efficient=str(i) + str(j))
		m, cores, simplex_map, tri = main.main(points, None)
		seg_map = main.window_seg(W,W,c,r,tri,simplex_map)
		master[i,j] = seg_map

# with open('master.pkl', 'rb') as f:
# 	master = pickle.load(f)

master_copy = copy.copy(master)

palette = 1000*np.random.random((len(cs),len(rs),np.max(master_copy)+2))

all_colors = np.zeros(np.shape(master_copy))
ms = np.zeros((len(cs),len(rs)))
for i in range(len(cs)):
	for j in range(len(rs)):
		all_colors[i,j] = palette[i,j,master_copy[i,j].astype(int) + 1]
		ms[i,j] = np.max(master_copy[i,j] + 2)

ms = ms.astype(int)

for i in range(len(cs)):
	for j in range(len(rs)):
		patch = all_colors[i,j]
		r = rs[0] + j*500*overlap
		c = cs[0] + i*500*overlap
		paste_in(patch, r, c, 0.7, overlap)

plt.imshow(canvas); plt.show()