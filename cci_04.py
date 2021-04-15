
class TreeNode:
	def __init__(self, val, left=None, right=None, parent=None, size=None):
		self.val = val
		self.left = left
		self.right = right
		self.parent = parent # unused in many problems
		self.size = size

	def __repr__(self): # mirroring Leetcode's convention on this:
		s = "["			# just print nodes in left->right, up->down order
		q = [self]
		while len(q) > 0:
			node = q.pop(0)
			if node:
				s += str(node.val) + ", "
				q.append(node.left)
				q.append(node.right)
			else:
				s += "_, "
		while s[-3] == '_': s = s[:-3]
		return s[:-2] + ']'

def one(G, S, E):
	# Use BFS. Would use bidirectional, but my graph is directed.
	# For bidirectional extension: Keep two queues and two visited
	# sets, and if a node being visited along one path turns out to
	# be in the other path's visited set, bingo.
	q = [S]
	visited = set()
	while len(q) > 0:
		node = q.pop(0)
		if node == E: return True

		for neighbor in G[node]:
			if neighbor not in visited:
				q.append(neighbor)
		visited.add(node)

	return False

G = {0: [1], # This is the graph drawn on page 106
	 1: [2],
	 2: [0,3],
	 3: [2],
	 4: [6],
	 5: [4],
	 6: [5]}
assert one(G, 3, 1)
assert not one(G, 0, 5)
assert one(G, 0, 3)
assert one(G, 4, 5)
assert not one(G, 6, 1)

def two(arr):
	# Recursive: middle of arr becomes this node's value, left and right
	# get the other array parts. Traverse in-order.
	if arr:
		mid = len(arr)//2
		return TreeNode(arr[mid], two(arr[:mid]), two(arr[mid+1:]))

tree = two(range(10))
assert str(tree) == "[5, 2, 8, 1, 4, 7, 9, 0, _, 3, _, 6]"
assert str(two(range(15))) == "[7, 3, 11, 1, 5, 9, 13, 0, 2, 4, 6, 8, 10, 12, 14]"

from cci_02 import Node

def three(root):
	lls = []
	ll_ends = []

	# Insight: You can use dfs here, because as long as we traverse left child first, nodes
	# in each layer will be visited left to right, and we can keep track of which linked
	# list to append to by just using an int to dscribe the level we're on.
	def dfs(node, level):
		if node:
			if not len(lls) > level:
				lls.append(Node(node.val))
				ll_ends.append(lls[level])
			else:
				ll_ends[level].next = Node(node.val)
				ll_ends[level] = ll_ends[level].next
			dfs(node.left, level+1)
			dfs(node.right, level+1)

	dfs(root, 0)
	return lls

assert str(three(tree)) == "[5->, 2->8->, 1->4->7->9->, 0->3->6->]"
tree.left.left.left = None
assert str(three(tree)) == "[5->, 2->8->, 1->4->7->9->, 3->6->]"

def four(root):

	def height(node):
		if node is None: return 1

		l = height(node.left)
		if l == 0: return 0 # shortcircuit -> don't run right side if imbalance already found
		r = height(node.right)
		return 1 + max(l, r) if l and r and abs(l-r) < 2 else 0

	return bool(height(root))

tree = TreeNode(1, TreeNode(2, TreeNode(4)), TreeNode(3))
assert four(tree)
tree.left.left.left = TreeNode(7)
assert not four(tree)

def five(root, inorder=False):
	if inorder: # whether to use inorder-based method
		val = [float('-inf')] # can't be a primative, or it won't be accessible
							# to inner function
		def inorder(node):
			if node:
				l = inorder(node.left)
				if val[0] > node.val: return False
				val[0] = node.val
				r = inorder(node.right)
				return l and r
			return True

		return inorder(root)

	else: # use ranges-based method
		def dfs(node, lo, hi):
			if node:
				if not lo <= node.val <= hi:
					return False
				l = dfs(node.left, lo, node.val)
				r = dfs(node.right, node.val+1, hi)
				return l and r
			return True

		return dfs(root, float('-inf'), float('inf'))

tree = two(range(10))
assert five(tree, inorder=True)
assert five(tree, inorder=False)
tree.right.left.right = TreeNode(100)
assert not five(tree, inorder=True)
assert not five(tree, inorder=False)

def six(node):
	# Either our node is the rightmost thing in a subtree, and we need the parent of the
	# parent of that subtree, or the node is itself a parent of a right subtree, in which
	# case we need the left-most thing in it.
	below = node.right
	while below and below.left:
		below = below.left
	if below: return below.val

	above = node.parent
	while above.parent and above.val < node.val:
		above = above.parent
	return above.val if above.val > node.val else None # return None if nothing to the right

tree = two(range(10))
def set_parent(node):
	if node.left:
		node.left.parent = node
		set_parent(node.left)
	if node.right:
		node.right.parent = node
		set_parent(node.right)
set_parent(tree)
assert six(tree) == 6 # the root holds 5
assert six(tree.left.right) == 5 # 5.left is 2, 2.right is 4. 4 has no right subtree
assert six(tree.left.right.left) == 4 # 4.left = 3, which is a leaf
assert six(tree.right.right) is None

from collections import defaultdict

def seven(projects, dependencies):
	Gforward = defaultdict(set)
	Gbackward = defaultdict(set)

	build_next = set(projects)
	for d in dependencies:
		Gforward[d[0]].add(d[1]) # f -> a, b
		Gbackward[d[1]].add(d[0]) # a -> f and b -> f

		build_next.discard(d[1]) # a, b get removed, because we know they have a dependency
	build_order = list(build_next)

	while len(Gforward) > 0:
		prev_step = build_next
		build_next = set()
		for node in prev_step:
			for child in Gforward[node]:
				# if length is 1, then the parent is going to be node, which we built at the
				# previous step. Otherwise, it's still possible building is okay, but all the
				# remaining dependencies need to have been accounted for by the previous step.
				if len(Gbackward[child]) == 1 or \
					Gbackward[child] | prev_step == prev_step:
					build_next.add(child)
				del Gbackward[child]
			del Gforward[node]
		if len(build_next) == 0: raise ValueError('unreachable nodes')
		build_order += list(build_next)

	return build_order

# second project is dependent on first
dependencies = [('a','d'), ('f','b'), ('b','d'), ('f','a'), ('d','c')]
projects = ['a','b','c','d','e','f']
order = seven(projects, dependencies)
for fr,to in dependencies:
	assert order.index(fr) < order.index(to)
dependencies = [('a','e'), ('e','d'), ('f','b'), ('b','d'), ('f','a'), ('d','c')]
order = seven(projects, dependencies)
for fr,to in dependencies:
	assert order.index(fr) < order.index(to)
dependencies = [('a','b'), ('b','c'), ('c','a')] # cycle makes this impossible
try:
	seven(projects, dependencies)
	assert False
except ValueError:
	assert True
# The example from the solutions. It passes!
projects = ['a','b','c','d','e','f','g']
dependencies = [('f','a'), ('f','b'), ('f','c'), ('c','a'), ('b','a'), ('b','e'), ('a','e'), ('d','g')]
order = seven(projects, dependencies)
for fr,to in dependencies:
	assert order.index(fr) < order.index(to)

def eight(root, a, b):
	# You could iterate upward using parent connections and store a set of seen nodes. First
	# go upward from a, then upward from b. When one of b's ancestors touches the set of a's
	# ancestors, that's the answer. But the problem specifies no extra storage. You could go
	# upward to find the depths of the respective nodes and then iterate up from the deeper
	# one by the difference and then both together until the pointers land on each other. But
	# I don't like relying on parent connections. So instead, return 0 if a node has neither
	# a nor b below it, 1 if it has one of them, and the node itself if its two subtrees each
	# contain 1 of the things we're looking for. Once a node is returned, keep sending it upward.
	def dfs(node, a, b):
		if node is None: return 0
		if node == a or node == b: return 1

		l = dfs(node.left, a, b)
		if type(l) is not int: return l # short circuit
		r = dfs(node.right, a, b)
		if type(r) is not int: return r

		if l == r == 1: return node
		return l + r

	p = dfs(root, a, b)
	return p if type(p) is not int else False

# Not necessarily a binary search tree
tree = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5, TreeNode(6), TreeNode(7))),
	TreeNode(3, TreeNode(8), TreeNode(9, right=TreeNode(10, right=TreeNode(11, TreeNode(12))))))
assert eight(tree, tree.left.left, tree.left.right.right).val == 2
assert eight(tree, tree.left.left, tree.right.right.right).val == 1
assert eight(tree, tree.right.left, tree.right.right.right).val == 3
assert not eight(tree, tree.right.left, TreeNode(100))

def nine(root):

	def inorder(node):
		if node:
			if node.left is None and node.right is None:
				return [[node.val]]

			left = inorder(node.left)
			right = inorder(node.right)
			p = []
			for l in left:
				for r in right:
					for o in interleave(l, r): # helper defined and tested below
						p.append([node.val] + o)
			return p

		else: return [[]]

	return inorder(root)

# nine needs this helper function
def interleave(a, b):
	# interleave l and r in all possible ways that preserve their orders
	if len(a) == 0: return [b]
	if len(b) == 0: return [a]
	v = []
	for x in interleave(a[1:], b):
		v.append([a[0]] + x)
	for x in interleave(a, b[1:]):
		v.append([b[0]] + x)
	return v

from math import comb

a = [1,2,3]
b = [4,5]
n = len(a)
m = len(b)
woven = interleave(a,b)
# https://math.stackexchange.com/questions/666288/number-of-ways-to-interleave-two-ordered-sequences
assert len(woven) == comb(n+m,m)
for weave in woven: # also make double sure a and b appear in each weave in order
	r = weave.index(a[0])
	for i in range(1,n):
		r_next = weave.index(a[i])
		assert r < r_next
		r = r_next
	r = weave.index(b[0])
	for i in range(1,m):
		r_next = weave.index(b[i])
		assert r < r_next
		r = r_next

little_tree = TreeNode(2, TreeNode(1), TreeNode(3, right=TreeNode(4)))
assert str(nine(little_tree)) == '[[2, 1, 3, 4], [2, 3, 1, 4], [2, 3, 4, 1]]'

def ten(t1, t2):

	def dfs(node, looking_for=None):
		if node:
			l = dfs(node.left, looking_for)
			if l == True: return l
			r = dfs(node.right, looking_for)
			if r == True: return r
			h = hash(str(node.val)) + l + r
			if looking_for == h: return True
			return h
		else: return 0

	return dfs(t1, dfs(t2)) == True

t2 = TreeNode(11, TreeNode(12))
assert ten(tree, t2)
t2 = TreeNode(5, TreeNode(6), TreeNode(7))
assert ten(tree, t2)
t2.right.val = 8
assert not ten(tree, t2)

import random

def eleven(root):
	# Very simple idea: Choose random number in [1, N], then go left or right depending
	# where that number falls against sizes of subtrees. If my left node says size of
	# its subtree is 5, and my right node says its size is 6, and I the parent take up
	# size 1, then if I draw anything in [1,5] I know it goes in left subtree, if I draw
	# 6 I know it's me the parent, and if I draw anything >6, I know it goes in right
	# subtree. But I have to be careful to say it's now the draw-|left tree|-1th thing
	# in the right subtree.
	def traverse(node, c): # O(log N)
		l = node.left.size if node.left is not None else 0
		if c == l + 1: return node
		elif c <= l: return traverse(node.left, c)
		else: return traverse(node.right, c-l-1)

	return traverse(root, random.randint(1, root.size))

# We need the tree and subtrees to keep track of their sizes to solve this problem
# in O(log N)
def set_size(node):
	if node:
		node.size = 1 + set_size(node.left) + set_size(node.right)
		return node.size
	return 0
set_size(tree)

# Testing randomness is tricky. I'm taking the "just run a bunch, and make sure it
# looks uniformish" approach
sim = [0]*tree.size
for i in range(100000):
	sim[eleven(tree).val-1] += 1
avg = sum(sim)/len(sim)
sim = [s/avg for s in sim]

for x in sim:
	assert abs(1-x) < 0.05

def twelve(root, s, o_n=True):

	# O(1) space, O(N log N) time
	if not o_n:
		def path_sum(node, a):
			if node:
				x = node.val + a
				lx = path_sum(node.left, x)
				rx = path_sum(node.right, x)
				return int(x == s) + lx + rx
			return 0

		def dfs(node):
			if node:
				l = dfs(node.left)
				r = dfs(node.right)
				return path_sum(node, 0) + l + r
			return 0

		return dfs(root)
	
	# O(n) space, O(n) time
	else:
		# Insight: Think of each path from the root to each leaf as an array. We basically want to
		# find the number of subarrays in each of these that sum to the target value, and then sum
		# those numbers. Example: [1,1,1,0,1,1], target=3. We can choose [1,1,1], [1,1,1,0], [1,1,0,1],
		# or [1,0,1,1], so the answer is 4 along this branch. There's a trick to find the number of
		# these quickly: Imagine a cumulative sum along this array: [0,1,2,3,3,4,5]. (There's an extra
		# 0 at the beginning.) We can now find whether a subarray from [i,j) matches the target by
		# subtracting: target = cumsum[j] - cumsum[i]. Rather than iterate all [i,j), we can do this
		# in O(n) similar to two-sum: If we pick a j, then we know we need cumsum[i] = cumsum[j] - target.
		# If we store cumsum as a map from value -> |occurrences of that value|, then we can figure out
		# how many subarrays of correct sum end at j in O(1) by just looking for how many i cumsum[i]
		# = cumsum[j] - target. We'll have to iterate j in [1,N] as we build up our traversal-path anyway,
		# which gives us an opportunity to fill the map. Caveat: cumsum[j] need to be removed from the
		# map too as dfs returns upward.
		d = defaultdict(int)
		d[0] = 1 # because you can always get a zero from the beginning

		def dfs(node, a):
			if node:
				a += node.val
				d[a] += 1 # one more subarray from 0->here has cumsum = a
				e = d[a-s] # the number of subpaths with sum s ending at this node
				l = dfs(node.left, a)
				r = dfs(node.right, a)
				d[a] -= 1 # this node is no longer considered, so one fewer subarrays with cumsum = a
				return e + l + r
			return 0

		return dfs(root, 0)

tree = TreeNode(1, TreeNode(0, TreeNode(1), TreeNode(1, TreeNode(0), TreeNode(1))),
	TreeNode(1, TreeNode(0), TreeNode(1, right=TreeNode(0, right=TreeNode(1, TreeNode(1))))))
assert twelve(tree, 4, o_n=False) == twelve(tree, 4, o_n=True) == 2
assert twelve(tree, 3, o_n=False) == twelve(tree, 3, o_n=True) == 5
assert twelve(tree, 2, o_n=False) == twelve(tree, 2, o_n=True) == 12
