# main.py
# Implementation of Vitaliy Kurlin's algorithm
# for the auto-completion of contours
# To try it out, uncomment the last four lines of code
# and run 'python main.py'

from scipy.spatial import Delaunay
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import time
import copy
import matplotlib as mpl
import grid

# Nodes in dual Delaunay complex
class ForestNode:
	def __init__(self,birth,simplex,i):
		self.birth = birth
		self.uplink = self
		self.height = 0
		self.live = []
		self.bar = -1
		self.simplex = simplex
		self.id = i

class Forest:
	def __init__(self, nodes):
		self.nodes = nodes
		self.links = 0

# Components in dual Delaunay complex
class MapNode:
	def __init__(self,ind,birth,death,core,heir):
		self.ind = ind
		self.birth = birth
		self.death = death
		self.core = core
		self.heir = heir
		self.supr = None

class Map:
	def __init__(self):
		self.comps = []
	def add_comp(self, c):
		self.comps.append(c)

# Plot persistence diagram
def plot_PD(mp, m):
	xs = [mapee.death for mapee in mp]
	ys = [mapee.birth for mapee in mp]
	s = np.array([30]*m + [1]*(len(mp) - m))
	plt.scatter(xs, ys, s)
	plt.plot(xs,xs)
	plt.show()

# Plot triangles
def plot_tris(verts, edge_colors, face_colors):
	new_verts = np.zeros((len(verts),3,2))
	new_verts[:,:,0] = verts[:,:,1]
	new_verts[:,:,1] = -verts[:,:,0]
	fig, axes = plt.subplots()
	if face_colors != None:
		coll = PolyCollection(new_verts, array = face_colors, cmap = mpl.cm.jet, edgecolors = edge_colors)
	else:
		coll = PolyCollection(new_verts, facecolors='white', cmap = mpl.cm.jet, edgecolors = edge_colors)
	axes.add_collection(coll)
	axes.autoscale_view()

# Plot different groups of triangles given by cores
def plot_cores(cloud, cores, text_points):
	n = len(cores)
	palette = [0.5 + 0.5*float(i)/n for i in range(n)]
	colors = []
	verts = np.zeros([0,3,2])
	all_tris = []
	for i in range(n):
		core_list = cores[i]
		tris = [c.simplex for c in core_list if c]
		tris = [tri for tri in tris if tri != None]
		new_verts = cloud[np.array(tris)]
		all_tris.append(new_verts)
		color = palette[i] *500
		colors += [color]*len(tris)
		verts = np.concatenate((verts,new_verts))
	plot_tris(verts, 'face', np.array(colors))
	plt.scatter(cloud[:,1], -cloud[:,0], s=2)
	if text_points != None:
		plt.scatter(text_points[:,0], -text_points[:,1], s=2)
	plt.show()
	return all_tris

# Main step of segmentation:
# Deduces which groups of simplices merge together
# Upon varying scale from infinity to zero
def build_map(cloud, debug=False):
	# Outputs sorted tuple of ids for a simplex's vertices
	def simplex_to_tuple(simplex):
		if simplex == "extra":
			return "extra"
		i, j, k = sorted(simplex)
		return (i,j,k)

	# Compute Delaunay triangulation
	tri = Delaunay(cloud)
	print "Triangulation done!"
	plot_tris(cloud[np.array(tri.simplices)], np.zeros(len(tri.simplices)), None)
	plt.show()

	K = len(tri.simplices)
	# Obtain list of edges
	edges = set()
	# Dict with keys: edge, values: 2 simplices containing edge
	edge_map = defaultdict(list)
	for simplex in tri.simplices:
		(i, j, k) = simplex_to_tuple(simplex)
		edges.add((i,j))
		edges.add((j,k))
		edges.add((i,k))
		edge_map[(i,j)].append(simplex)
		edge_map[(j,k)].append(simplex)
		edge_map[(i,k)].append(simplex)

	# Those with just a single neighbor must have
	# the extra vertex as a neighbor
	for edge, neighbors in edge_map.iteritems():
		if len(neighbors) < 2:
			edge_map[edge].append('extra')

	# Square distance between points in tuple tup
	def edge_squared(tup):
		x1, y1 = cloud[tup[0]]
		x2, y2 = cloud[tup[1]]
		return (x1-x2)**2 + (y1-y2)**2

	# Sort edges of del in decreasing order of length
	edges_squared = {edge: edge_squared(edge) for edge in edges}
	edges = sorted(list(edges), key=edges_squared.get,reverse=True)
	# Initialize forest
	def get_tri_edges_squared(simplex):
		(i,j,k) = simplex_to_tuple(simplex)
		a = edges_squared[(i,j)]
		b = edges_squared[(j,k)]
		c = edges_squared[(i,k)]
		return np.array([a, b, c])

	# Computes circumradius
	def circumradius(simplex):
		mat = np.ones((3,3))
		mat[:,:-1] = cloud[simplex]
		area = np.abs(np.linalg.det(mat))/2
		a,b,c = np.sqrt(get_tri_edges_squared(simplex))
		R = a*b*c/(4*area)
		return R

	# Determines whether a triangle is acute
	def is_acute(simplex, debug=False):
		a,b,c = get_tri_edges_squared(simplex)
		check1 = c - a - b
		check2 = b - a - c
		check3 = a - b - c
		if debug:
			print "checks: "
			print check1, check2, check3
		return (check1 < 0.) & (check2 < 0.) & (check3 < 0.)

	# Chases up tree to find root of tree containing v
	def get_root(v):
		while v.uplink != v:
			v = v.uplink
		return v

	# Keys: simplex, Values: dual forest node
	nodes = {simplex_to_tuple(simplex): (ForestNode(circumradius(simplex),simplex,i) if is_acute(simplex) else ForestNode(0,simplex,i)) for i, simplex in enumerate(tri.simplices)}
	for node in nodes.values():
		if node.birth > 0:
			node.live = [node]
	# "Extra" node is the node corresponding to the exterior
	nodes['extra'] = ForestNode(np.infty,None,len(nodes))
	nodes['extra'].live = [nodes['extra']]
	forest = Forest(nodes.values())
	# Main loop
	edge_counter = 0
	map_node_counter = 0
	mp = Map()
	round_counter = 0
	# Keep going as long as forest is disconnected
	while forest.links < K:
		if debug:
			print "ROUND " + str(round_counter) + "\n"
		round_counter += 1
		if debug:
			for node in sorted(nodes.values(), key=lambda x: x.id):
				print str(node.id) + " root: " + str(get_root(node).id), " bar: " + str(node.bar) + " live: " + str([live.id for live in node.live])
		# Take next longest edge
		edge = edges[edge_counter]
		edge_counter += 1
		length = np.sqrt(edges_squared[edge])
		# Scale alpha = length(edge)/2
		alpha = length/2
		# Get two triangles sharing this edge
		Tu, Tv = edge_map[edge]
		u = nodes[simplex_to_tuple(Tu)]; v = nodes[simplex_to_tuple(Tv)]
		if debug:
			print "Linking.... " + str(u.id) + " and " + str(v.id)
		root_u = get_root(u)
		root_v = get_root(v)
		# Case 1
		# If roots agree, then do nothing
		if root_u == root_v:
			if debug:
				print "Case 1"
			continue
		if root_u.birth > root_v.birth:
			# Swap u,v so we can assume u's component is younger
			temp = u; u = v; v = temp
			temp = root_u; root_u = root_v; root_v = temp
		if root_u.birth == 0 and root_v.birth > 0:
			# Case 2
			# One root was not born yet, merge into alive tree of v
			if debug:
				print "Case 2: " + str(root_u.id) + " linked to " + str(root_v.id)
			u.uplink = root_v
			u.birth = root_v.birth
			if root_v.height == 1:
				root_v.height == 2
			root_v.live.append(u)
			# If bar exists, v is already dead, so u will be as well
			if v.bar != -1:
				if debug:
					print "bar exists"
				u.bar = v.bar
			else:
				u.live = [u]
			forest.links += 1
			continue
		# Create new MapNode
		birth = root_u.birth
		death = alpha
		core = root_u.live
		heir = root_v.live[0]
		if debug:
			print "BAR SET"
		for w in root_u.live:
			w.bar = map_node_counter
		c = MapNode(map_node_counter,birth,death,core,heir)
		map_node_counter += 1
		mp.add_comp(c)

		if root_u.height <= root_v.height:
			# Case 3
			# Both u and v were already alive, so u must be merged into v
			# Because u's tree is shorter, u's tree becomes a subtree of v's
			if debug:
				print "Case 3: " + str(root_u.id) + " linked to " + str(root_v.id)
			root_u.uplink = root_v
			if root_u.height == root_v.height:
				root_v.height += 1
		else:
			# Case 4
			# Both u and v were already alive, so u must be merged into v
			# Because u's tree is taller, v's tree becomes a subtree of u's
			if debug:
				print "Case 4: " + str(root_v.id) + " linked to " + str(root_u.id)
			root_v.uplink = root_u
			root_u.live = copy.copy(root_v.live)
		forest.links += 1

	# Create final component representing exterior
	last = nodes['extra']
	root_last = get_root(last)
	birth = root_last.birth
	death = alpha
	core = root_last.live
	heir = None
	for w in root_last.live:
		w.bar = map_node_counter
	c = MapNode(map_node_counter,birth,death,core,heir)
	mp.add_comp(c)

	cores = [mapee.core for mapee in mp.comps]
	plot_cores(cloud, cores, None)
	return mp, K, tri

# Given Map of components and their neighboring relations,
# Output segmentation of underlying space of cloud
def segmentation(cloud, mp, K, text_points = None, plot = True, m=None, ignore=False):
	mp_unsorted = mp.comps
	N = len(mp_unsorted)
	pers_dict = {c: c.birth - c.death for c in mp_unsorted}
	# Sort components by decreasining order of persistence
	mp_sorted = sorted(mp_unsorted, key=pers_dict.get, reverse=True)
	# Want to find maximum gap in sorted list of persistences
	persists = sorted(pers_dict.values(), reverse=True)
	if not m:
		max_gap = 0; max_gap_i = 0
		for i in range(1,N-1):
			gap = persists[i] - persists[i+1]
			if gap > max_gap:
				max_gap = gap
				max_gap_i = i
		m = max_gap_i + 1
	# Plot persistence diagram for visualization purposes
	plot_PD(mp_sorted, m)

	# Compute mapping of element indices before and after sorting
	pre_sort_map = sorted(range(len(mp_unsorted)), key=lambda i:pers_dict[mp_unsorted[i]], reverse = True)
	sort_map = [0]*len(pre_sort_map)
	for i,x in enumerate(pre_sort_map):
		sort_map[x] = i

	# Compute superior components
	pre_suprs = [c.heir.bar if c.heir else None for c in mp_sorted]
	suprs = [sort_map[pre_supr] if pre_supr else None for pre_supr in pre_suprs]
	for i in range(0,N):
		if suprs[i]:
			mp_sorted[i].supr = mp_sorted[suprs[i]]
		else:
			mp_sorted[i].supr = mp_sorted[0]

	# Get final destinations of each region upon merging
	final_dests = {i:i for i in range(m)}
	for i in range(m,N):
		j = sort_map[mp_sorted[i].supr.ind]
		if j < i:
			final_dests[i] = final_dests[j]
		else:
			while j > i:
				j = sort_map[mp_sorted[j].supr.ind]
			final_dests[i] = final_dests[j]

	# Tells us which core each triangle is in
	simplex_map = {}
	for i in range(m):
		core_id = final_dests[i]
		c = mp_sorted[i]
		for node in c.core:
			simplex_map[node.id] = core_id

	# Compile cores of most persistent components
	for i in range(m,N):
		core_id = final_dests[i]
		dest = mp_sorted[core_id]
		c = mp_sorted[i]
		for node in c.core:
			simplex_map[node.id] = core_id
		if dest.core:
			dest.core += c.core
		else:
			dest.core = [c.core]

	cores = [c.core for c in mp_sorted[0:m]]
	if plot:
		tris = plot_cores(cloud, cores, text_points)
	else:
		tris = None

	return m, cores, tris, simplex_map

# If we also have the bit matrix for the image, output segmentation in bit matrix
def window_seg(w, h, offsetx, offsety, tri, simplex_map):
	seg_map = -np.ones((h,w))
	for y in range(h):
		for x in range(w):
			tri_id = tri.find_simplex((x+offsetx,y+offsety))[()]
			if tri_id != -1:
				core_id = simplex_map[tri_id]
				seg_map[y,x] = core_id
	return seg_map

def main(pts, text_points, debug=False, window=None):
	mp, K, tri = build_map(pts,debug)
	print "Built map!"
	m, cores, tris, simplex_map = segmentation(pts, mp, K, text_points)
	return m, cores, simplex_map, tri

# pts = grid.grid(3,0.001,3000)
# mp, K, nodes = build_map(pts,False)
# m, cores, simplex_map, tri = segmentation(pts, mp, K)