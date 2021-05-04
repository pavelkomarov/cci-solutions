
def one(A, B):
	# A and B are composed of bits, so we can do a ripple-add if we keep an extra carry bit and
	# draw a K-map to figure out what the logic should be. Time for some ASCII art!
	# Call the addition result bit x, call the carry bit c, and call bits from A and B, a and b.
	#
	#      \ ab                                    \ ab
	#     c \ _00__01__11__10                     c \ _00__01__11__10
	#      0 | 0 | 1 | 0 | 1 |                     0 | 0 | 0 | 1 | 0 |
	#        |---------------| = x                   |---------------| = c'
	#      1 | 1 | 0 | 1 | 0 |                     1 | 0 | 1 | 1 | 1 |
	#        '---------------'                       '---------------'
	#
	# Now we draw boxes and do some boolean algebra. For x we unfortunately can't do any better
	# than single-element boxes, but it actually ends up simplifying down a lot:
	#
	#		x = !a!bc + a!b!c + abc + !ab!c = !a(!bc + b!c) + a(bc + !b!c) = !a(b^c) + a!(b^c)
	#		  = a^(b^c) = a^b^c
	#
	# 		c' = ab + !abc + a!bc = ab + (!ab + a!b)c = ab + (a^b)c
	X = 0

	c = 0
	for i in range(32):
		a = (A >> i) & 1
		b = (B >> i) & 1

		x = a^b^c
		X |= (x << i)
		
		c = (a & b) | ((a ^ b) & c)

	# In most languages that would be sufficient, but because python interprets bit strings with a 1
	# in the sign-bit position as really big positive numbers instead of negative numbers, yet still
	# represents negative numbers under the hood in 2s complement, then if the answer is negative
	# you have to do some funny bit manipulation to get back to the proper negative result:
	# Step 1: take 2s complement of the wrong big positive number to get it as a negative number.
	# Step 2: logical-and that with a bitmask so we chop off the improper big bits. We now have a
	#		  small positive number.
	# Step 3: take 2s complement again to get the proper negative result.
	return X if not (X >> 31) else ~((~X + 1) & 0xffffffff) + 1

assert one(100,14) == 114
assert one(-100,14) == -86

from random import randint
from collections import defaultdict
import numpy

def two(arr):
	"""Use the Fisher-Yates algorithm"""
	for i in range(len(arr)-1, 0, -1): # don't need to go all the way to 0, because swapping with self is null op
		k = randint(0, i)

		t = arr[k]
		arr[k] = arr[i]
		arr[i] = t

d = defaultdict(int)
for i in range(12000):
	arr = list(range(5))
	two(arr)
	d[str(arr)] += 1

assert len(d) == 120
v = list(d.values())
assert numpy.mean(v) == 100 # average number of times each appears is 100
assert numpy.std(v) < 15 # with a not-too-large variance

def three(arr, m, with_dictionary=False):
	n = len(arr)

	if with_dictionary: # The trick here is *very* similar to the randomrange problem Two Sigma asked me
		d = {} # store "corruptions" of the range
		s = []
		for i in range(n-1, n-m-1, -1):
			k = randint(0, i)

			if k not in d: # we've hit an "uncorrupted" location, so k is k
				s.append(arr[k])
				if k != i: # if k == i, then we can't pick this k again next round, so no worries
					if i not in d:
						d[k] = i
					else:
						d[k] = d[i]
			else: # we've hit a corruption, so k -> some former, larger i
				s.append(arr[d[k]])
				d[k] = i

	else: # This can actually be done without extra memory. Say we have a random set of size m for the
		# subproblem up to but not including the ith element. We want to solve the problem including the
		# ith element. The probability any of the 0th..i-1th members are currently in the set is m/i.
		# Now we want the probability that any of the 0th..ith members are in the set to be m/(i+1). We'll
		# choose a random number k in 0..i. There is a 1/(i+1) chance we select any given number. There
		# is an m/(i+1) chance that we select k < m. If we flip the ith number with the kth if k < m, then
		# each of the m numbers in the set has a m/i * (1 - 1/(i+1)) probability of remaining in the set =
		# m/i * ((i+1)/(i+1) - 1/(i+1)) = m/i * i/(i+1) = m/(i+1). Perfect.
		s = arr[:m]

		for i in range(m,n):
			k = randint(0,i)

			if k < m:
				s[k] = arr[i]

	return set(s)

arr = list(range(10))
for wd in [True, False]:
	d = defaultdict(int)
	for i in range(12000):
		m = sorted(list(three(arr, 3, with_dictionary=wd)))
		d[str(m)] += 1

	assert len(d) == 120 # 10 choose 3 is 120 different sets
	v = list(d.values())
	assert numpy.mean(v) == 100 # average number of times each appears is 100
	assert numpy.std(v) < 15

def four(arr):
	# The key to this one is tossing numbers that we know can't be the one we're looking for.
	# That way we have n, n/2, n/4, n/8 ... numbers to consider at each iteration, and the
	# problem shrinks to O(n) work (the sum of the series).
	#
	# If n is even, then 0..n is an odd number of numbers
	#	-> there should be one more 0 in the 1s place than there are 0s in the 1s place.
	# If n is odd, then 0..n is an even number of numbers
	#	-> there should be as many 0s in the 1s place as 1s in the 1s place
	# But we're missing a number.
	# If it's an even number, then we'll have one fewer 0s than expected
	# If it's an odd number, we'll have one fewer 1s than expected
	# set or don't set the corresponding bit in the missing number to account for the missing bit.
	#
	# Once we figure out whether missing is odd or even, remove all the numbers from the set it couldn't
	# be. Because none of the numbers we remove are the missing one, we don't lose information. For the
	# 2s place: If the number is divisible by 4, we have a 0 there, if only divisible by 2, we have a 1
	# there, and instead of the sequence 0 0 1 1 0 0 1 1 ... in that bit, we removed half of them, so
	# it's now 0, 1, 0, 1, ... and the same arguments for bit-frequency apply for these n//2 numbers.
	m = 0
	nums = set(arr)

	for i in range(32):
		n = len(nums)
		if n == 0: break

		ones = sum((x >> i) & 1 for x in nums)
		zeros = n - ones

		if (n % 2 == 0 and zeros > ones + 1) or (n % 2 == 1 and zeros > ones):
			# we're missing a number with a 1 at the ith location
			m |= (1 << i)
			# remove all numbers from set that have a 0 at the ith location
			nums = set(x for x in nums if (x >> i) & 1)
		else: # otherwise the number has a 0 at the ith location
			# we don't need to manipulate `missing`, but we do need to remove all numbers from
			# the set that have a 1 at the ith location
			nums = set(x for x in nums if not (x >> i) & 1)

	return m

for n in [110,13]:
	for missing in [3,6,13]:
		arr = list(range(missing)) + list(range(missing+1, n+1))
		assert four(arr) == missing

def five(arr):
	"""My idea here was to keep a running sum as a list, incrementing every time I see a number,
	decrementing every time I see a letter (or vice versa, doesn't matter), then look at the result.
	I have to be careful to start from list=[0], and then append list[-1] +-1. I end up with something
	like: [0, -1, -2, -1, -2, -1, 0, -1, 0, -1, -2, -1, -2, -1, 0, -1, -2, -3, -4, -5, -6]

	This immediately suggests an algorithm to me: If I start at a certain running sum and then get back to
	the same sum, then the sum of all the numbers in between is zero, which means an even number of letters
	and numbers. I need to keep the earliest occurrence of each new running sum so I can base off it if
	need be, and if I come across a sum I've seen before, then I know I can get the length of that distance
	by subtracting my current index from the index where I first saw that sum (my "base").
	"""
	d = {} # store bases
	r = 0 # running sum
	d[r] = -1 # my running sum is zero before I start running along indices. First index is 0, so just before is -1.
	m = 0 # length of largest evenly alphanumeric string I've seen so far
	#ll = [r]

	for i,a in enumerate(arr):
		if 48 <= ord(a) <= 57: # numeric
			r += 1
		elif 97 <= ord(a) <= 122: # alphabetical
			r -= 1
		#ll.append(r)

		if r in d:
			m = max(m, i - d[r])
		else:
			d[r] = i

	#print(ll)
	return m

assert five("abc345") == 6
assert five("a3b4c5d") == 6
assert five("i0ryjlfjq1cdwbbwzvij") == 2
assert five("oxxalkyr4fkh11szzlfo") == 6
assert five("7tl91m1xbjfye951mnqb") == 8
assert five("c8bo5qdzevfj2u700u3v") == 10
assert five("oy6c12a5al8w57gyjlal") == 14

def six(n, brute_force=False):
	twos = 0

	if brute_force: # for testing against
		for i in range(n+1):
			for c in str(i):
				if c == '2':
					twos += 1
	else:
		"""
		There's a recursive pattern here:

		For the 1s: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 -> 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 -> sums to 1
		For the 10s: 0s, 10s, 20s, 30s, 40s, 50s, 60s, 70s, 80s, 90s
			-> 1, 1, 11, 1, 1, 1, 1, 1, 1, 1 -> sums to 20
		For the 100s: 0s, 100s, 200s, 300s, 400s, 500s, 600s, 700s, 800s, 900s
			-> 20, 20, 120, 20, 20, 20, 20, 20, 20, 20 -> sums to 300
		For the 1000s: 0ks, 1ks, 2ks, 3ks, 4ks, 5ks, 6ks, 7ks, 8ks, 9ks
			-> 300, 300, 1300, 300, 300, 300, 300, 300, 300, 300 -> sums to 4000

		In general each 10^k in the number contributes k*10^(k-1) 2s, unless its the third 10^k we're
		counting, in which case you add in an extra 10^k 2s.

		So in 3030 we get 300 2s from each of the 1000s, an extra 1000 2s from the 2000s, one 2 from each
		of the three 10s, and an extra 10 2s for the 20s, bringing the total to 1913.

		The trouble is when we have 2s in number itself. For every 2 found, we know that all the numbers
		with smaller or equal value in digits of lesser significance than this digit can contribute an extra
		2, which only allows us to consider up to n.

		So say we have 2222. We get 300 2s from the first two thousands, which covers 0..999, 1000..1999.
		But now for 2000..2222 We have to account for the extra 2 in the thousands place for all subsequent
		numbers. The number of these twos is just the cardinality of [0, 222] = 223 = n % 1000 + 1. You
		can likewise do this ones-place upward rather than thousands-place downward, because you'll find
		all the same sets to account for.
		"""
		h = n
		k = 0
		while h > 0:
			h, d = divmod(h, 10)

			twos += d*k*10**(k-1) # d is in the (10^k)s place -> each contribute k*10^(k-1) 2s

			if d > 2: # if we're beyond the 2s, then just add in all extra 10^k 2s
				twos += 10**k
			elif d == 2: # if we're *in* the 2s, then we know 2(k 0s)..2(rest of n) each contribute
				twos += n % (10**k) + 1	# an extra 2. n % 10^k + 1 is the number of things in this set.

			k += 1

	return int(twos) # I think ** causes int -> float under some strange condition

assert six(25, brute_force=True) == six(25, brute_force=False) == 9
assert six(3030, brute_force=True) == six(3030, brute_force=False) == 1913
assert six(2222, brute_force=True) == six(2222, brute_force=False) == 892

def seven(names, synonyms, union_find=False):
	if union_find:
		# Essentially each name is a graph node, and each synonym relationship adds an edge. I need
		# the connected connected components in this graph to get canonical names. I'm using a Union Find.
		connected = []
		for name1,name2 in synonyms:

			e1 = None # the sets the two names belong to
			e2 = None
			for i,x in enumerate(connected): # This is my favoite way to do a union-find in python
				if name1 in x: e1 = i
				if name2 in x: e2 = i

			if e1 == e2 and e1 is not None: continue

			elif e1 is None and e2 is None: # then create new set in the union-find
				connected.append(set([name1, name2]))

			elif e1 is None or e2 is None: # add one to the other
				if e1 is None: connected[e2].add(name1)
				else: connected[e1].add(name2)

			else: # they both exist, but in different sets, so merge
				connected[e1] |= connected[e2]
				del connected[e2]

		# I'd like to map from all members of each connected component to some canonical name
		d = {}
		for component in connected:
			label = component.pop()
			for name in component:
				d[name] = label

	else:
		# There is a more optimal way to find the connected components in a graph. With a union
		# find you have to keep iterating the components and merging sets, and it ends up O(n log n)
		# in the worst case. If instead you build a graph out of all your nodes and edges and then DFS
		# it, keeping a visited set, you can get connected components in O(V + E) time.
		G = {}
		for name1,name2 in synonyms:
			if name1 in G:
				G[name1].add(name2)
			else:
				G[name1] = set([name2])

			if name2 in G:
				G[name2].add(name1)
			else:
				G[name2] = set([name1])

		d = {} # functions as the visited set
		def dfs(node, label):
			d[node] = label
			for neighbor in G[name]:
				if neighbor not in d:
					dfs(neighbor, label)

		for name in G:
			if name not in d:
				dfs(name, name)

	# Now use the known labels to total up frequencies
	totals = defaultdict(int)
	for name,freq in names:
		if name in d:
			name = d[name]
		totals[name] += freq

	return list(totals.items())

names = [('John', 15), ('Jon', 12), ('Chris', 13), ('Kris', 4), ('Christopher', 19)]
synonyms = [('John', 'Jon'), ('John', 'Johnny'), ('Chris', 'Kris'), ('Chris', 'Christopher')]
for uf in [True, False]:
	for name,total in seven(names, synonyms, union_find=uf):
		if name in ['John', 'Jon', 'Johnny']:
			assert total == 27
		elif name in ['Chris', 'Kris', 'Christopher']:
			assert total == 36
		else:
			assert False

def eight(heightweights, recurse=False):
	"""Because we know height and weight must both be sorted in the final subsequence, we can partially
	solve the problem by sorting the heightweights list by height or weight at the beginning. Then we
	know the subsequence that's the right answer is ordered in the sorted array, and we only have to
	search *after* (or before) each i to find potential js.
	"""
	heightweights = sorted(heightweights)

	if recurse:
		"""This is very similar to 8.13, the stacking boxes problem. Basically we have to put things on top
		of each other, but this is only allowed via certain conditions.

		Say I have a recursive function which assumes person i is at the top. Person j right below them has
		to be taller and heavier, and I can iterate through the people looking for who is taller and heavier.
		Assume each person j is the top of their own tower, and see how tall that tower can be. Select the
		sub-tower with the most people in it, and then stack person i atop, and return.
		
		Memoize so the recursion isn't stupid.
		"""
		memo = {}
		def recurse(i):
			if i in memo: return memo[i]
			longest = 0
			t = []
			h1, w1 = heightweights[i]

			for j,(h2,w2) in enumerate(heightweights[i+1:]):
				if h2 > h1 and w2 > w1:
					t_j = recurse(i+1+j)
					if len(t_j) > longest:
						t = [x for x in t_j]
						longest = len(t_j)

			t.insert(0, heightweights[i])
			memo[i] = t
			return t

		longest = []
		for i in range(len(heightweights)): # I'm pretty sure this ends up O(n^2)
			t_i = recurse(i)
			if len(t_i) > len(longest):
				longest = t_i

		return longest

	else:
		"""Alternative: If you sort by height or weight, then the problem reduces to finding the longest
		increasing subsequence in the other variable. This is a classic problem with O(n^2) and O(n log n)
		solutions.

		Briefly: In O(n^2) you iterate through the array finding out how long the longest increasing
		subsequence that ends *at the ith location* is. Figuring this out invovles iterating backwards to
		find the best jth location to build off of.

		In the O(n log n) solution you build up a fanciful sequence by placing numbers *where they would
		go* in a potential sequence that could be created with them. This is best illustrated with an
		example: Input: [0, 8, 4, 12, 2]. sequence is [0], then [0, 8] because 8 can only build off the 0,
		[0, 4] because 4 can still build off 0 but is lower than 8 and therefore more likely to allow
		subsequent numbers to be greater than it, [0, 4, 12], because 12 can build off 4, [0, 2, 12] because
		2 can build off 0 and is even lower than 4. The real kicker here is that we can figure out where to
		place numbers in this sequence we're building in O(log n) time.

		In order to recover the actual sequence, we have to keep track of ancestors.
		"""
		nums = [x[1] for x in heightweights] # get the weights
		ancestors = {nums[0]:(None,0)} # so I can recover the subsequence, map num -> ancestor num (next key), index

		seq = [nums[0]] # the thing I'll be binary searching through
		for i in range(1, len(nums)):
			lo = 0
			hi = len(seq) - 1
			
			while lo <= hi:
				mid = (lo + hi) // 2
				
				if nums[i] > seq[mid]:
					lo = mid + 1
				else: # favor moving hi if ==, so lo ends up at the == location if one exists
					hi = mid - 1
				   
			# lo ends up at number just above (where to insert nums[i]), hi ends up at number just below (the ancestor)
			if lo == len(seq):
				seq.append(nums[i])
			else:
				seq[lo] = nums[i]

			ancestors[nums[i]] = (seq[hi],i) if hi >= 0 else (None,i)

		# recover the LCI (raw indexes), and use it to index heightweights to get the subsequence
		lci = []
		anc = seq[-1]
		while anc:
			anc, i = ancestors[anc]
			lci.insert(0, heightweights[i])

		return lci

heightweights = [(65,100), (70,150), (56,90), (75,190), (40, 100), (60,95), (68,110)]
assert eight(heightweights, recurse=True) == eight(heightweights, recurse=False) == \
	[(56, 90), (60, 95), (65, 100), (68, 110), (70, 150), (75, 190)]












