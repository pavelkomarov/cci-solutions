
import numpy

class BitVector: # a bit vector class, worth implementing
	def __init__(self, l):
		self.v = numpy.zeros(-(-l//32), dtype=int)
		self.l = l

	def __getitem__(self, i):
		if i >= self.l: raise ValueError("index out of range")
		div, mod = divmod(i, 32)
		return (self.v[div] >> mod) & 1

	def __setitem__(self, i, x):
		if i >= self.l: raise ValueError("index out of range")
		div, mod = divmod(i, 32)
		self.v[div] = self.v[div] | (1 << mod)

	def __repr__(self):
		s = "["
		for x in self.v:
			for i in range(32):
				s += "1" if (x >> i) & 1 else "0"
			s += '\n '
		return s[:-2] + "]"

bv = BitVector(100)
assert len(bv.v) == 4 # it takes 4*32 > 100
for i in range(100):
	assert bv[i] == 0
	bv[i] = 1
	assert bv[i] == 1
try:
	bv[101]
	assert False
except ValueError:
	assert True
assert str(bv) == "[11111111111111111111111111111111\n" + \
				  " 11111111111111111111111111111111\n" + \
				  " 11111111111111111111111111111111\n" + \
				  " 11110000000000000000000000000000]" # 32*3 = 96, so 4 1s left over for last entry

def one(A, B):
	i = len(A) - 1 # end location in A
	a = len(A) - len(B) - 1 # location of last filled value in A
	b = len(B) - 1 # location of last value in B

	while a >= 0 and b >= 0:
		if A[a] >= B[b]:
			A[i] = A[a]
			a -= 1
		else:
			A[i] = B[b]
			b -= 1
		i -= 1
	# if a >= 0, then those items are already placed
	while b >= 0: # But if b >= 0, then all of A was shifted, 
		A[i] = B[b] # and we need to copy over the rest of B
		i -= 1
		b -= 1

A = list(range(1,11,2)) + [0]*5
B = list(range(2,11,2))
one(A, B)
assert A == list(range(1, 11))

from collections import defaultdict, Counter

def two(S):
	buckets = defaultdict(list)
	for s in S:
		c = tuple(sorted(Counter(s).items())) # you could sort the string itself, alternatively
		buckets[c].append(s)

	a = []
	for b,l in buckets.items():
		a += l

	return a

S = ["aaa", "aab", "aba", "cba", "baa", "abc", "fff", "efg", "acb"]
assert two(S) == ['aaa', 'aab', 'aba', 'baa', 'cba', 'abc', 'acb', 'fff', 'efg']

def three(A, t):
	# The key is that if we divide the array in two, we'll get a half that's normally ordered
	# and a half that's still wrapped. If it's still wrapped, then we have the same problem
	# over again, so recurse. If normally ordered, then a standard binary search will do it.
	# If we make our binary search also recursive, then we can just do the two in one function.
	def recurse(lo, hi):
		if lo > hi: return # terminating contition if something not found

		mid = (lo + hi)//2
		if A[mid] == t: return mid # terminating condition if found

		if A[mid] < A[hi]: # then the right is normally ordered
			if A[mid] < t <= A[hi]: # we're looking for something in an ordered array, so just binary search
				return recurse(mid + 1, hi)
			else: # the only place it could be is the left side of the array, which will have the same wrapped structure
				return recurse(lo, mid - 1) # In the event we're operating on a totally ordered piece, then
					# this condition also properly searches the ordered left side if t < A[mid].
		
		elif A[mid] > A[lo]: # then the left is normally ordered
			if A[mid] > t >= A[lo]:
				return recurse(lo, mid - 1) # standard binary search
			else:
				return recurse(mid + 1, hi) # doing the opposite

		else: # It's impossible for both A[mid] < A[lo] and A[mid] > A[hi], because one of the sides is
			# ordered. However, we can have A[mid] == A[lo], indicating duplicate values on the left, or
			# A[mid] == A[hi], indicating duplicate values on the right, or A[lo] == A[mid] == A[hi],
			# indicating mostly duplicate values through the whole array.
			i = None
			if A[mid] == A[hi]: # search the left side, because the right is goofy
				i = recurse(lo, mid - 1)
			if A[mid] == A[lo] and not i: # search the right side, because left is goofy and didn't give us an i
				i = recurse(mid + 1, hi)
			return i

	return recurse(0, len(A)-1)
	# All the conditions in this function are very tricky to think through. It's clearly just a modified
	# binary search, and all the magic is in getting the conditionals right. I ended up having to read
	# through the first part of the solution a couple times for it to start to make sense. In particular,
	# how to tell which half is correctly ordered, and what to do in the cases of duplicates, are difficult.

A = [15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14]
for i,t in enumerate(A):
	assert three(A, t) == i
assert three(A, 11) is None
A = [70, 75, 17, 18, 30, 31, 35, 50, 60]
assert three(A, 18) == 3
A = [24, 25, 26, 27, 30, 31, 13, 18, 23]
assert three(A, 18) == 7
A = [30, 30, 30, 30, 30, 31, 13, 18, 23] # duplicates at left
assert three(A, 18) == 7
A = [70, 75, 17, 18, 30, 30, 30, 30, 30] # duplicates at right
assert three(A, 18) == 3
A = [30, 30, 30, 18, 30, 30, 30, 30, 30] # nearly all duplicates (both)
assert three(A, 18) == 3
A = [30, 30, 30, 30, 30, 30, 30, 18, 30]
assert three(A, 18) == 7

def four(L, x):
	lo = 0
	hi = 1
	# it takes O(log n) to find the end of the array
	e = L.elementAt(hi)
	while e < x and e != -1:
		hi <<= 1 # multiply by 2
		e = L.elementAt(hi)

	while lo <= hi: # takes O(log n) again to find x
		mid = (lo + hi)//2
		e = L.elementAt(mid)
		if e == -1: # Our range can (and likely will at first) run off the end,
			hi = mid - 1 # so move hi down accordingly
		elif e == x:
			return mid
		elif e < x:
			lo = mid + 1
		else:
			hi = mid - 1

class Listy:
	def __init__(self):
		self.l = []

	def elementAt(self, i):
		return self.l[i] if i < len(self.l) else -1

L = Listy()
L.l += range(100)
for i in range(100):
	assert four(L, i) == i
assert four(L, 1000) is None

def five(S, s):
	lo = 0
	hi = len(S) - 1

	while lo <= hi:
		left = True # assume there is a nonempty string to the left
		mid = (lo + hi)//2
		while S[mid] == "" and mid >= 0: mid -= 1 # my binary search modification
		
		if mid == -1: # there was nothing prior to our thing
			left = False # there was no nonempty string to the left
			mid = (lo + hi)//2
			while S[mid] == "": mid += 1 # these loops makes it in worst case, if totally sparse, O(n),
										# but it's more like O(log n) if there are enough entries
		if S[mid] == s: return mid

		og_mid = (lo + hi)//2
		if s < S[mid]: # search left:
			hi = mid - 1 if left else og_mid - 1 # go as far left as possible (cut out known empty strings)
		else: # search right
			lo = og_mid + 1 if left else mid + 1 # go as far right as possible

S = ["at", "", "", "", "ball", "", "", "car", "", "", "dad", "", ""]
assert five(S, "at") == 0
assert five(S, "ball") == 4
assert five(S, "car") == 7
assert five(S, "dad") == 10
S = ["", "", "", "", "", "", "", "car", "", "", "dad", "", ""]
assert five(S, "ball") is None
assert five(S, "car") == 7
assert five(S, "dad") == 10
S = ["at", "", "", "", "ball", "", "", "", "", "", "", "", ""]
assert five(S, "at") == 0
assert five(S, "ball") == 4
assert five(S, "car") is None

# Six
# I'd do a merge sort on the lines. First just take little chunks of 5 lines or so and sort them,
# then merge, merge, merge, merge up to the full 20 GB. We used to do this for alphabetizing graded
# tests when I was a CS TA. You'd need auxiliary space for this, not a ton if you delete info from
# the original file as you copy and mergesort it elsewhere. You couldn't really do a quicksort so
# easily, because lines might have different lengths, so swapping them around each other would
# result in overlaps unless you shift *everything* in between, which is gross.

def seven(nums):
	# The way to do this if you have tons of memory is to just create a gigantic bit vector to keep track
	# of what has been seen. Iterate back through looking for an unset bit, and there's your missing
	# number. An alternative if you have limited memory is to do this over smaller ranges, but if we have
	# 2^31 possible nonnegative ints and only a little over 2^26 bits in 10 MB, then we might have to do
	# 32 passes in order to cover the full range. Though we could get lucky and find a missing number much
	# earlier, 1 billion is about 2^30, so we could very well end up deep in, and at that point, may as
	# well sort the numbers and look for a gap, because log(1 billion) is 29.897, which is < 32. A clever
	# alternative from the solutions is to establish ranges of numbers and iterate through the array
	# counting how many fall on each range. Since numbers must be unique, we can tell that a range which
	# hasn't completely filled must have something missing. Then we can go back and make a bit vector just
	# for that range, iterate the big array again, tossing things outside the range, setting bits within,
	# then iterate the bit vector looking for a 0 just like before.

	# Let's pretend I have 10 kilobytes instead of 10 megabytes. I can fit a little over 2^16 bits.
	# That's 2^14 bytes, so I can have that many buckets. 2^31/2^14 = 2^17, which is what I expect
	# my buckets to fill to if there's no empty space on a range.
	n_buckets = 2**14
	bucket_size = 2**31 // n_buckets

	buckets = numpy.zeros(n_buckets, dtype=int)
	for n in nums:
		buckets[n//bucket_size] += 1

	for i,b in enumerate(buckets):
		if b < bucket_size: break

	bv = BitVector(bucket_size)
	ibsz = i*bucket_size
	ibszz = ibsz + bucket_size
	for n in nums:
		if ibsz <= n < ibszz:
			bv[n-ibsz] = 1

	for z in range(bucket_size):
		if bv[z] == 0: break

	return ibsz + z

missing = 500005
nums = numpy.arange(1000000) # do this in megabytes rather than gigabytes for time.
nums[missing] = 1000000 # at the 500 thousanth location we get a missing value
numpy.random.shuffle(nums)

assert seven(nums) == missing

def eight(nums):
	# Another bit vector. A kilobyte is 1024*8 bits, and we have 4 of them, so that gets us over
	# 32000. We can mark when things have been seen, and when we see them again, print, or store
	# to file, or, if few enough, keep in that little tiny extra bit of memory.
	bv = BitVector(N)
	dups = []
	for n in nums:
		if bv[n]: dups.append(n)
		else: bv[n] = 1

	return dups

from random import randint, shuffle

N = 32000
nums = list(range(N))
randomz = [randint(0,N-1) for i in range(30)]
nums += randomz
shuffle(nums)

assert sorted(eight(nums)) == sorted(randomz)

def nine(matrix, x):
	# Any rows/cols that have start > x or end < x can't contain x.
	# If we binary search along the top row for x and find no x, but find the first element > x, then
	# x can't be in that column or any column to its right. Likewise if we search along the bottom
	# row and find the last element < x, then x can't be in that column or any column to its left.
	# We can do the same with leftmost and rightmost columns, eliminating rows. Then repeat.
	n, m = matrix.shape
	ri = 0 # index of first row we're searching
	rj = n - 1 # index of last row we're searching
	ci = 0 # same for columns
	cj = m - 1

	def bin_search(ndx, lo, hi, row=True): # row controls whether we're searching a row or a col
		while lo <= hi:
			mid = (lo + hi)//2
			e = matrix[ndx, mid] if row else matrix[mid, ndx]
			if e == x: return True, mid

			if e < x:
				lo = mid + 1
			else:
				hi = mid - 1

		# at the end, if x isn't found, hi and lo will have overstepped each other. hi will be on
		# the element just below x, and lo will be on the element just above x. I'm choosing a
		# convention to always return the just-larger location, the place x would go if it were inserted
		return False, lo

	# Alternate searching rows and cols, zeroing in on x's location. This took me ages to beat all the
	# little bugs out. Very tricky problem.
	while True:
		found, w = bin_search(ri, ci, cj, row=True) # search the uppermost row to limit the last col
		if found: return ri, w
		cj = w - 1 # since I'm returning the location above the last feasible col, the last one is that-1
		if cj < 0: return # we've eliminated all columns, so the number isn't here

		found, ci = bin_search(rj, ci, cj, row=True) # search the lowest row to limit first col
		if found: return rj, ci
		if ci >= m: return # we've eliminated all columns, so the number isn't here

		found, w = bin_search(ci, ri, rj, row=False) # search leftmost col to limit last row
		if found: return w, ci
		rj = w - 1 # since I'm returning the location above the last feasible row, the last one is that-1
		if rj < 0: return # we've eliminated all rows, so the number isn't here
		
		found, ri = bin_search(cj, ri, rj, row=False) # search rightmost col to limit first row
		if found: return ri, cj
		if ri >= n: return # we've eliminated all rows, so the number isn't here

m = numpy.array([[ 0, 3, 6, 9, 12, 18, 21],
				 [ 2, 4, 6,10, 14, 20, 25],
				 [ 8, 9,10,11, 15, 27, 30],
				 [13,20,28,30, 90,100,102]])
for i in range(m.shape[0]):
	for j in range(m.shape[1]):
		a = nine(m, m[i,j])
		assert m[a] == m[i,j]
assert nine(m, 1000) is None
assert nine(m, -1) is None

from cci_04 import TreeNode

class Ten:
	def __init__(self):
		self.tree = None

	def track(self, x):
		# insert node in self.tree, updating subtree sizes on the way down
		node = self.tree
		parent = None
		while node:
			parent = node
			parent.size += 1 # increment size
			if x <= node.val:
				node = node.left
			else:
				node = node.right

		if parent:
			if x <= parent.val:
				parent.left = TreeNode(x, size=1)
			else:
				parent.right = TreeNode(x, size=1)
		else:
			self.tree = TreeNode(x, size=1)

	def get_rank(self, x):
		# Every time we go rightward, we gain |left subtree| + 1 nodes to our left. Every
		# time we go leftward, we gain no new nodes to the left. When we run in to the number
		# we're looking for, it also might have a left subtree of its own, so add its size to
		# the number of leftward nodes accumulated on the way down, and that's the rank.
		node = self.tree
		par_l = 0 # count of ancestors and their left subtrees
		while node:
			if x < node.val: # going left, so we haven't gained anything to left
				node = node.left
			else:
				sub_l = 0 if node.left is None else node.left.size 

				if node.val == x: return par_l + sub_l

				par_l += sub_l + 1 # add left subtree of parent and the parent itself
				node = node.right

ranker = Ten()
stream = [5, 1, 4, 4, 5, 9, 7, 13, 3]
for i in stream:
	ranker.track(i)
assert ranker.get_rank(1) == 0
assert ranker.get_rank(3) == 1
assert ranker.get_rank(4) == 3
assert ranker.get_rank(13) == 8
assert ranker.get_rank(7) == 6
assert ranker.get_rank(100) is None

def eleven(A):
	# First idea: if the array is sorted, you just swap pairs. Only works if there
	# aren't duplicate elements in a pair.
	#A = sorted(A)
	#for i in range(2, len(A), 2):
	#	t = A[i]
	#	A[i] = A[i-1]
	#	A[i-1] = t
	#return A

	# New idea: Take the array in trios, moving the largest element to the middle.
	# only works if there isn't a duplicate trio. The problem doesn't specify whether there
	# can be duplicates, but the example has one. I think the problem statement should
	# explicitly exclude them. Otherwise you might even have degenerate cases without a solution.
	for i in range(2, len(A), 2):
		a = A[i-2]
		b = A[i-1]
		c = A[i]
		if a > c and a > b:
			A[i-2] = b
			A[i-1] = a
		elif c > a and c > b:
			A[i] = b
			A[i-1] = c
		# otherwise we're gucci already, because b is largest

	# It's possible the last element wasn't included in a trio, so swap if needed
	if len(A) % 2 == 0 and A[-1] < A[-2]: A[-2:] = reversed(A[-2:])

for i,A in enumerate([[5,3,1,2,3], [1,2,3,4,5,6,7,8,9,0], list(range(100))]):
	if i == 2: shuffle(A)
	eleven(A)
	for i in range(1, len(A)):
		if i % 2 == 1:
			assert A[i] > A[i-1]
		else:
			assert A[i] < A[i-1]
