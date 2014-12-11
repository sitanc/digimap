from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Image
import random

k = 200
e = 0.1
omega = 0.2

class RipsNode:
	def __init__(self,coords,neighbors,curv):
		self.coords = coords
		self.neighbors = neighbors
		self.curv = curv
		self.i = -1

class RipsEdge:
	def __init__(self,node1,node1,curv):
		self.ends = [node1, node1]
		self.curv = curv
		self.i = -1

class RipsSimplex:
	def __init__(self,comps,curv):
		self.comps = comps
		self.curv = curv
		self.marked = False
		self.dim = len(self.comps)

def get_cloud(name):
	im = Image.open('test_images/'+ name +'.png').convert('1')
	w, h = im.size
	x = im.load()

	cloud = []
	for i in range(0,w):
		for j in range(0,h):
			if x[i,j] < 255:
				pert_x = 2*random.random()-1
				pert_y = 2*random.random()-1
				cloud.append(np.array([i+pert_x,j+pert_y]))

	cloud = np.array(cloud)
	# plt.scatter(cloud[:,0], cloud[:,1],s=0.0001); plt.show()
	return cloud

# TODO: Remove points whose eigenvalue ratio is greater than 0.25
def fibers(cloud):
	nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(cloud)
	_, indices = nbrs.kneighbors(cloud)
	fs_vects = []
	for i, p in enumerate(cloud):
		p_neighbors = cloud[indices[i,:],:]
		avg = sum(p_neighbors)/k
		M = p_neighbors - np.tile(avg, (len(p_neighbors),1))
		cov = np.dot(np.transpose(M),M)
		eigvals, eigvecs = linalg.eig(cov)
		fiber = eigvecs[np.argmax(eigvals)]
		fs_vects.append(fiber)
	fs_vects = np.array(fs_vects)
	angles = np.angle(fs_vects[:,1]*1j + fs_vects[:,0])
	# angles[angles >0] = angles[angles>0] - 2*np.pi
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(cloud[:,0], cloud[:,1], angles,s=0.0001)
	ax.scatter(cloud[:,0], cloud[:,1], angles + np.pi,s=0.0001)
	# ax.scatter(cloud[:,1], cloud[:,0], -angles,s=0.0001)
	plt.show()
	fs = np.append(cloud, angles.reshape(len(cloud),1), axis=1)
	return fs

def curvatures(cloud, fs):
	def curv(p):
		raise Error("NEED TO IMPLEMENT")


def build_rips(fs, cloud, omega):
	# p,q both have shape (1,3)
	def dist(p,q):
		p[3] %= np.pi*2; p[3] *= omega
		q[3] %= np.pi*2; q[3] *= omega
		return np.sqrt(np.sum((p - q)**2))
	curvs = curvatures(cloud, fs)
	num_fs = len(fs)
	sort_map = sorted(range(num_fs), key=lambda i: curvs[i], reverse=True)
	points = points[sort_map]
	nodes = [None]*num_fs
	edges = []
	for i in reversed(range(num_fs)):
		p = points[i,:]
		neighbs = []
		for j in range(i+1,num_fs):
			q = points[j,:]
			if dist(p,q) < e:
				neighbs.append(nodes[j])
		node = RipsNode(p, neighbs,curv)
		nodes[i] = node
		for neighb in neighbs:
			edge = RipsEdge(node,neighb,curvs[i])
			edges.append(edge)
	return nodes, edges

def barcode(nodes, edges):
	def remove_pivot_rows(simplex):
		k = simplex.dim
		if k == 2:
			d = [comp in simplex.comps if comp.marked]
			coeffs = [(-1)**(i+1) for i in range(2)]
			a, b = simplex.comps[0].i, simplex.comps[1].i
			while max(coeffs) != 1 != min(coeffs):
				i = simplex.comps[0].i
				if T[i] == []:
					break
				q = T_coeffs[i][i]


	simplices = []
	for node in nodes:
		simplex = RipsSimplex([node],node.curv,False)
		simplices.append(simplex)
	for edge in edges:
		simplex = RipsSimplex(edge.ends,edge.curv,False)
		simplices.append(simplex)
	simplices = sorted(simplices, key=lambda x: (x.curv, x.dim))
	for i, simplex in enumerate(simplices):
		if simplex.dim == 1:
			simplex.comps[0].i = i
	T = [None]*len(simplices)
	T_coeffs = [None]*len(simplices)
	for k in range(2):
