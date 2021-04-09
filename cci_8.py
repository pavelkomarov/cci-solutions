
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

def five(a,b):
	if a > b: return five(b,a)
	if a == 1: return b

	# drive smaller number to 1
	v = five(a >> 1, b)
	if a % 2 == 0:
		return v+v
	else:
		return v+v + b

assert five(278, 597) == five(597, 278) == 278*597

def six(t1, t2, t3):
	c = [0]
	
	def move(n, fr, to, e):
		c[0] += 1
		# move n rings from the `fr` stack to the `to` stack using the `e` stack
		# as the intermediary
		if n == 1:
			to.append(fr.pop())
		else:
			move(n-1, fr, e, to) # move one fewer from `fr` to `e` using `to` as intermediary
			to.append(fr.pop()) # move the nth single largest bottom ring from `fr` to `to`. intermediary not relevant
			move(n-1, e, to, fr) # move the n-1 that we moved to `e` earlier from `e` to `to` using `fr` as intermediary
		
		# Make sure all towers are decreasing at the end of each move call
		for tower in [t1,t2,t3]:
			for i in range(1,len(tower)):
				assert tower[i] < tower[i-1]

	move(len(t1), t1, t3, t2)
	return c[0] # I wanted to know how big this recursion gets

N = 5
t1 = list(range(N,0,-1))
t2 = []
t3 = []
assert six(t1,t2,t3) == (1 << N) - 1 # turns out you make 2^N - 1 moves
assert t1 == []
assert t2 == []
assert t3 == list(range(N,0,-1))

def seven(s):
	# The strategy here is insertion: abcde -> a inserted at all possible locations
	# in all permutations of bcde, bcde -> b inserted at all possible locations in
	# all permutations of cde ... e -> e itself.
	if len(s) == 1: return [s]

	a = s[0]
	R = seven(s[1:])
	p = []
	for r in R:
		p += [r[:i] + a + r[i:] for i in range(len(r)+1)]
	return p
	
from math import factorial, prod

s = "abcde"
perms = seven(s)
assert len(perms) == factorial(len(s))
assert len(set(perms)) == factorial(len(s))
for p in perms: assert set(p) == set(s)

from collections import Counter

def eight(s):
	# Rather than insertion, this one works better with an alternative recurrence:
	# abcde -> a + all permutations of bcde, b + all permutations of acde, etc. This
	# is equally legitimate, because the a + perms part gives all terms where a begins
	# a permutation, b + for b, etc., and the terms where a is at different locations
	# will be taken care of in terms from other parts.
	def recurse(c):
		if len(c) == 1: # base case
			for a in c: break
			return [a*c[a]] # So something like "aaaaaaaaa" finishes immediately

		p = []
		for a in c:
			new_c = {k:v for k,v in c.items() if k != a}
			if c[a] > 1: new_c[a] = c[a]-1
			p += [a + r for r in recurse(new_c)]
		
		return p

	return recurse(Counter(s))

s = "ABCAAC"
perms = eight(s)
assert len(perms) == len(set(perms))
c = Counter(s)
assert len(perms) == factorial(len(s))//prod([factorial(v) for v in c.values()])

def nine(n, build=True):
	if not build:
		# less efficient way, because there are some duplicates between ()+r and r+(),
		# which we have to loop through and remove
		if n == 1: return ["()"]
		R = nine(n-1)
		p = ["(" + r + ")" for r in R]
		p += ["()" + r for r in R]
		p += [r + "()" for r in R]
		return set(p) # looping through and removing duplicates, basically

	else:
		def recurse(prefix, l, r):
			# More efficient way. l is number of ( remaining to insert, r is number of )
			# remaining to insert. If l > 0, then we can insert a (. If l < r, meaning
			# more l inserted so far, then we can insert a ).
			if l == 0 and r == 0: return [prefix]

			p = []
			if l > 0:
				p += recurse(prefix + "(", l-1, r)
			if l < r:
				p += recurse(prefix + ")", l, r-1)
			return p

		return recurse("", n, n)

parens = nine(5)
assert len(parens) == len(set(parens))
for p in parens:
	c = Counter(p)
	assert c[')'] == c['(']

def ten(screen, p, c):
	k = screen[p]

	def recurse(i,j):
		screen[i,j] = c

		for y,x in [(i+1,j),(i,j+1),(i-1,j),(i,j-1)]:
			if 0 <= x < screen.shape[0] and 0 <= y < screen.shape[1] and \
				screen[y,x] == k:
				recurse(y,x)

	recurse(*p)

screen = numpy.zeros((10,10))
screen[:,4] = 1
ten(screen, (2,4), 9)
assert numpy.sum(screen) == 90
assert numpy.all(screen[:,4] == 9)

def eleven(n):
	# This is similar to the stairs, question 1, except 1,5 and 5,1 are considered the
	# same. The recurrence ends up being very different:
	# T[n, {1,5,10,25}] = T[n, {1,5,10}] -> using no quarters to get n
	#					+ T[n-25, {1,5,10}] -> using 1 quarter and having n-25 left
	#				 ...+ T[n-25x, {1,5,10}] -> x is the largest int such that n-25x > 0
	# This gives us all possible ways of introducing quarters. We repeat for dimes and
	# nickels. But when we get to only pennies we know there's only one way to build
	# up to whatever the residual value is. So the base case is T[n, {1}] = 1.
	memo = {} # repeat calls with the same inputs are possible.
	coins = [1,5,10,25]
	
	def recurse(n, ndx):
		# I was originally sending ever-smaller copies of coins down, but better
		# to keep one coins array and just index it.
		if ndx == 0: return 1 # when we get down to pennies, only one way
		elif (n, ndx) in memo: return memo[(n, ndx)]

		s = 0
		c = coins[ndx]
		m = n
		while m >= 0:
			s += recurse(m, ndx-1)
			m -= c
		memo[(n, ndx)] = s
		return s

	return recurse(n, len(coins)-1)

assert eleven(25) == 13

def twelve(n):
	# encode queen placement as a list of 8 numbers in [0,7], so the ith
	# queen is at (i, Q[i])
	def recurse(placed):
		if len(placed) == n: return [placed]
		p = []
		i = len(placed)
		for c in range(n): # next queen will be at (i,c)
			# if this choice of `c` doesn't interfere with any already-placed
			# queen, then we won't break, and we'll reach the else
			for j,q in enumerate(placed): # formerly placed is at (j,q)
				if c == q or i+c == j+q or i-j == c-q:
					break
			else:
				p += recurse(placed + [c])
		return p

	return recurse([])

queens = twelve(8)
# https://en.wikipedia.org/wiki/Eight_queens_puzzle is known to have 92 solutions
assert len(queens) == len(set(tuple(q) for q in queens)) == 92
for q in queens: assert len(set(q)) == 8

def thirteen(boxes):
	# sort boxes by decreasing width, so they're at least kind of in order: earlier boxes
	# must come before later boxes.
	boxes = sorted(boxes) # python sorts tuples really nicely on all indices, similar to
	boxes.reverse() 	# alphabetizing, so (5,5,4) comes before (5,5,5), and then reverse
	memo = {} # ith box -> the max height possible with this box as bottom

	def recurse(i): # The ith box is *included* in some subsequence that's being selected
		# from the boxes, and each new call picks the best jth next box based on which, when
		# serving as base for the remaining tower, maximizes additional height. When we run
		# up against a next box that will fit, assume it's correct, and compute forward. 
		if i in memo: return memo[i]
		b1 = boxes[i]
		m = b1[2]
		for j,b2 in enumerate(boxes[i+1:]):
			if b2[0] < b1[0] and b2[1] < b1[1] and b2[2] < b1[2]:
				m = max(m, b1[2] + recurse(i+1 + j))
		memo[i] = m
		return m

	# pick the best first box, and the recurrence will take care of picking the next-best
	return max([recurse(i) for i in range(len(boxes))])

boxes = [(100,100,1),(10,10,10),(5,5,4),(4,4,6),(50,40,9),(1,1,20),(9,9,9),(1,1,1)]
assert thirteen(boxes) == 26
boxes[-1] = (1,5,1)
assert thirteen(boxes) == 25
boxes[5] = (1,1,100)
assert thirteen(boxes) == 100

def fourteen(s, result, save_space=True):
	# expression alternates symbol,operator,symbol,operator,symbol. We can split around any
	# operator.
	if not save_space:
		# This is the way I came up with initially. It wasn't clear to me exactly how to sum
		# up the counts, so I just recursively evaluated everything. The trouble with that is
		# you have to keep a bunch of lists of boolean values around, which feels unnecessary.
		def evaluate(s):
			if len(s) == 1: return [False] if s == '0' else [True]

			p = []
			for i in range(1,len(s),2):
				A = evaluate(s[:i])
				B = evaluate(s[i+1:])

				for a in A:
					for b in B:
						if s[i] == '&':
							p.append(a and b)
						elif s[i] == '|':
							p.append(a or b)
						elif s[i] == '^':
							p.append(a ^ b)

			return p

		return sum(e==result for e in evaluate(s))

	else:
		# Had to peek at the solutions to get this insight: Every sub-expression, will evaluate
		# to some number of trues and some number of falses based on where you put parentheses.
		#
		# If you think of all these trues and falses for expression A strung out along the edge
		# of a truth table, and you put the trues and falses from a second expression, B, along
		# the other edge, then the table has U := |A| x |B| elements, and how you fill it depends
		# on the operator.
		#
		# If it's a &, then only the quadrant where A is true and B is true will be true. This
		# will have |A = true| x |B = true| values.
		#
		# If it's a |, then only the quadrant where A is false and B is false will be false.
		# So you'll have U - |A = false| x |B = false| trues.
		# 
		# And for ^ there are |A = true| x |B = false| + |A = false| x |B = true| trues.
		#
		# And then of course for the number of falses it's just the complement U - T, where T
		# is the number of trues for whatever operator.
		#
		# The brilliant thing is we don't have to compute these massive tables, because if we
		# know |A = true|, |A = false|, |B = true|, |B = false|, we can get everything we need!
		memo = {} # there will be repeated calls, especially for smaller substrings

		def recurse(s): # return (num false, num true)
			if len(s) == 1: return (0,1) if s == '1' else (1,0)
			if s in memo: return memo[s]

			p = [0,0]
			for i in range(1,len(s),2):
				nA, A = recurse(s[:i])
				nB, B = recurse(s[i+1:])

				# we're almost taking a cartesian product of possibilities
				U = (A + nA)*(B + nB)

				if s[i] == '&': T = A*B # both have to be true
				elif s[i] == '|': T = U - nA*nB # either or both can be true
				elif s[i] == '^': T = U - A*B - nA*nB # have to be opposite

				p[1] += T
				p[0] += U - T

			memo[s] = p
			return p

		return recurse(s)[int(result)]

assert fourteen("1^0|0|1", False, save_space=False) == fourteen("1^0|0|1", False) == 2
assert fourteen("0&0&0&1^1|0", True, save_space=False) == fourteen("0&0&0&1^1|0", True) == 10
