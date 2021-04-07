
def one(n):
	# Very similar to the knight dialer. Basically you can get to the ith step by
	# stepping by 1, 2, or 3, and within each of those the last action is the same,
	# so the number of paths to i through step i-1 = the number of ways to get to i-1.
	# Same for the other two. And because they're distinct options, we can just add
	# them together: ways(i) = ways(i-1) + ways(i-2) + ways(i-3). What's the base case?
	# There is 1 way to start out on the ground ("step 0"), and there are 0 ways to
	# be at steps -1 and -2, because they don't exist.
	a = 0
	b = 0
	c = 1

	for i in range(n):
		d = a+b+c
		a = b
		b = c
		c = d

	return d

assert one(12) == 927
assert not one(36) > 2**31-1
assert one(37) > 2**31-1

def two(grid):
	table = numpy.ones(grid.shape)*float('inf')
	table[0,0] = 0

	for i in range(1, grid.shape[0]):
		if grid[i,0] == 0 and table[i-1,0] < float('inf'):
			table[i,0] = table[i-1,0] + 1
	for j in range(1, grid.shape[1]):
		if grid[0,j] == 0 and table[0,j-1] < float('inf'):
				table[0,j] = table[0,j-1] + 1

	for i in range(1, grid.shape[0]):
		for j in range(1, grid.shape[1]):
			if grid[i,j] == 0:
				table[i,j] = min(-1 if i==0 else table[i-1,j], -1 if j==0 else table[i,j-1]) + 1

	if table[i,j] == float('inf'): return None

	path = []
	while i > 0 and j > 0:
		if table[i-1,j] < table[i,j-1]:
			path.append('down')
			i -= 1
		else:
			path.append('right')
			j -= 1
	while i > 0:
		path.append('down')
		i -= 1
	while j > 0:
		path.append('right')
		j -= 1

	path.reverse()
	return path

import numpy
grid = numpy.zeros((4,5))
grid[1,1] = 1
grid[1,2] = 1
grid[2,0] = 1
assert two(grid) == ['right', 'right', 'right', 'down', 'down', 'down', 'right']
grid[0,1] = 1
assert two(grid) is None

def three(A):

	def recurse(l, r):
		if r < l: return None

		m = (r+l)//2
		if A[m] == m: return m
		
		# For non-duplicate case we have that if A[m] > m, then we know the magic index has to be
		# to the left, and for A[m] < m, the magic index has to be to the right. This doesn't
		# hold when duplicates are allowed. Instead we always have to search both right and left.
		# But, if A[m] already = some big value, then when we go right, we know that between the
		# mth and A[m]th entries in the array we can't succeed, because the only case that works
		# would be A[m] repeats all the way to the A[m]th thing. Similarly, if A[m] = some small
		# value, then we can't succeed in the left direction before the A[m]th location, in which
		# case the A[m]th value repeats all the way to that location.
		#
		# So in the non-repetition case we get A[m] > m -> search [l, m-1]; A[m] < m -> search [m+1,r]
		# In the repetition case we get: search [l,m-1] and [m+1,r], truncating the correct one at
		# A[m], which looks like either [l,A[m]] and [m+1,r] or [l,m-1] and [A[m],r]. Note it's
		# possible for one of those ranges to be empty (A[m] < l or A[m] > r), and in the non-
		# repetition case, one of them always will be.
		#
		# Therefore if we let our ranges be [l, min(m-1, A[m])] and [max(m+1, A[m]), r] and always
		# query both, the algorithm works for both the non-duplicates and duplicates case.
		a = recurse(l, min(m-1, A[m]))
		if a is not None: return a
		return recurse(max(m+1, A[m]), r)

	return recurse(0, len(A)-1)

A = [-100,-90,-80,3,5,100,200]
assert three(A) == 3
A = [-100,-90,-80,2,4,100,200]
assert three(A) == 4
A = [-100,-90,-80,-70,-60,-50,6]
assert three(A) == 6
A = [0,40,50,60,70,80,90]
assert three(A) == 0
A = [-100,-90,-80,2,6,6,7,8,11,11,11,11,20,40]
assert three(A) == 11

def four(s):
	if len(s) == 0: return [set()]

	a = s.pop()
	B = four(s)
	C = [b.copy() for b in B]
	for c in C:
		c.add(a)
	return B + C

assert len(four(set([1,2,3,4]))) == 16 # sum i = 0 to 4 (4 choose i) = 16
assert str(four(set([1,2]))) == "[set(), {2}, {1}, {1, 2}]"

#def five()








