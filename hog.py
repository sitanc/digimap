import numpy as np
from scipy import signal
from params import *

# convolve with DoG
def DoG(sigma):
	w = 2*np.floor(np.ceil(7*sigma)/2) + 1
	w = int(w)
	yy, xx = np.meshgrid(range(-(w-1)/2, (w+1)/2), range(-(w-1)/2,(w+1)/2))
	f = (1/sigma**4)*np.exp(-(xx**2 + yy**2)/(2*sigma**2))
	Dx = -xx*f
	Dy = -yy*f
	return Dx, Dy

Dx, Dy = DoG(2.)

def filter_window(window):
	xnew = signal.convolve2d(window, Dx, mode='same')
	ynew = signal.convolve2d(window, Dy, mode='same')
	return xnew, ynew

def get_magnitudes(window, xnew=None, ynew=None):
	if xnew == None or ynew == None:
		xnew, ynew = filter_window(window)
	magnitudes = np.sqrt(xnew**2 + ynew**2)
	# Cut off parts that are too small
	magnitudes[magnitudes<thres] = 0
	return magnitudes

def get_hog(window, thres=thres, n_bins=n_bins, hog_size=hog_size, rand=True):
	b = len(window); a = len(window[0,:])
	def hog_hist(x,y, n_bins, hog_size):
		hog_rad = (hog_size - 1)/2
		x0 = max(0,x-hog_rad); x1 = min(b,x+hog_rad+1); xs = range(x0,x1)
		y0 = max(0,y-hog_rad); y1 = min(a,y+hog_rad+1); ys = range(y0,y1)
		# print x0, x1, y0, y1
		yy, xx = np.meshgrid(xs, ys)
		orien_patch = orientations[yy,xx]
		bins_patch = ((orien_patch / 360. * n_bins) % n_bins).astype(int)
		mag_patch = magnitudes[yy,xx]
		return [len(bins_patch[(bins_patch == i) & (mag_patch != 0)]) for i in range(n_bins)]
	xnew, ynew = filter_window(window)
	magnitudes = get_magnitudes(window, xnew, ynew)
	# Compute orientations of all gradient vectors
	orientations = np.arctan(ynew/xnew)/np.pi*180
	orientations[xnew < 0] = 180 + orientations[xnew < 0]
	orientations[(xnew > 0) & (ynew < 0)] = 360 + orientations[(xnew > 0) & (ynew < 0)]
	hog = np.zeros((b,a,n_bins))
	for i in range(b):
		print i
		for j in range(a):
			hog[i,j,:] = hog_hist(i,j, n_bins, hog_size)
	return hog

# intensities = np.zeros((b,a))
# for i in range(b):
# 	print i
# 	for j in range(a):
# 		intensities[i,j] = sum(hog[i,j,:] > 0)

# test_intensities = copy.copy(intensities)
# test_intensities[(intensities<5) | (magnitudes == 0)] = 0
# plt.imshow(test_intensities, cmap='gray'); plt.show()

