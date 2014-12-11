# get_data.py
# Retrieves and processes training data from
# www.public.asu.edu/~bli24/icassp2013.html

import os
import hog
import Image
import json
import numpy as np
import random
from params import *

# Read in training data id's
ids = []
for name in os.listdir('DataSet/UpdatedGtMask'):
	ids.append(name[:name.find('_')])

# name for file with labels
def mask_file(i):
	return 'DataSet/UpdatedGtMask/'+i+'_updatedgtmask.png'

# name for file with pixels
def im_file(i):
	return 'DataSet/FloormapImages/'+i+'.jpg'

# converts image to matrix
# if mask is true, converts image to bitmask
def extract_mat(im, mask=False):
	im_mat = im.load()
	w, h = im.size
	if mask:
		mat = np.zeros((w,h), dtype=bool)
	else:
		mat = np.zeros((w,h))
	for x in range(w):
		for y in range(h):
			if mask:
				mat[x,y] = int(im_mat[x,y] > 0)
			else:
				mat[x,y] = im_mat[x,y]
	return mat

# Processes raw training data
# Creates HoG features for each image
def main():
	for i in ids[15:]:
		if os.path.getsize(mask_file(i)) < 4000:
			mask = Image.open(mask_file(i))
			im = Image.open(im_file(i)).convert('1')
			w, h = im.size
			labels = extract_mat(mask,True)
			im_mat = extract_mat(im)
			with open('training/'+i+'_lab','wb') as f:
				json.dump(labels.tolist(), f)
			print i+" labels written"
			hg = hog.get_hog(im_mat, thres, n_bins, hog_size)
			with open('training/'+i+'_hog','wb') as f:
				json.dump(hg.tolist(), f)
			print i+" hog written"

# Organizes files written by main() into
# a file of all HoG features of images and a file of binary
# labels of those images as text or not
def compile_all():
	all_labels = []
	all_hogs = np.zeros((0,n_bins))
	training_ids = []
	for name in os.listdir('training/'):
		training_ids.append(name[:name.find('_')])

	training_ids = list(set(training_ids))
	training_ids = [i for i in training_ids if os.path.getsize('training/'+i+'_hog')/1000. < 100000]
	for training_id in training_ids:
		lab_file_name = training_id + '_lab'
		hog_file_name = training_id + '_hog'
		lab_file = open('training/'+lab_file_name, 'rb')
		hog_file = open('training/'+hog_file_name, 'rb')
		labels = np.array(json.load(lab_file)).astype(int)
		hogs = np.array(json.load(hog_file))
		w,h = np.shape(labels)
		coin_tosses = np.random.random((w,h))
		sample_lab = (coin_tosses < 0.75) & (labels == 1)
		sample_unl = (coin_tosses < 0.30) & (labels == 0)
		sample = sample_lab | sample_unl
		some_hogs = hogs[sample,:]
		some_labs = labels[sample]
		print np.shape(some_hogs)
		all_hogs = np.concatenate((all_hogs, some_hogs),axis=0)
		all_labels += list(some_labs.flatten())
		lab_file.close()
		hog_file.close()

	with open('training/all_labs', 'wb') as out:
		json.dump(all_labels, out)
	with open('training/all_hogs', 'wb') as out:
		json.dump(all_hogs.tolist(), out)

