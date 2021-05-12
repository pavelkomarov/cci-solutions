
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

def nine(k, brute_force=False):
	if brute_force:
		"""The insight here is that the kth number will be at most 3^a * 5^b * 7^c, where the sum of
		a, b, c can't exceed k. Iterate a whole bunch of possibilities O(k^3), then sort O(k^3 log(k^3)),
		for a final run time of O(k^3 log(k)) (because the exponent inside the log is just a constant *
		the log). Pretty bad, but useful for checking correctness against other methods.
		"""
		l = []
		for a in range(k+1):
			for b in range(k+1-a):
				for c in range(k+1-a-b):
					l.append(3**a * 5**b * 7**c)
					
		return sorted(l)[k-1]
	else:
		"""Idea, arrived at by pondering and looking at the hints: keep our master list = [] to start,
		and keep three other lists 3s, 5s, 7s all also empty to start. We add 1 to the master list, and
		add 3*1 to the 3s, 5*1 to the 5s, 7*1 to the 7s. Then the next-best thing to put in the list is
		the min of the front of the 3s, 5s, and 7s queues, which will be 3. Push 3*3, 3*5, 3*7 on the
		queues, and repeat. Note that because we only need to return the kth factor, the master list can
		be notional.

		Note that we'll run in to duplicate factors here, because 3*5 is the same as 5*3, for example.
		One way around this is just to store them and iterate past them if a duplicate is encountered,
		which is how I solved it at first, but there is a space-saving trick:

		If the min value, x, comes from the 3s, then push on to all three queues. If it comes from the 5s,
		then 3*x will have already been seen as 5*(x*3/5), because we've visited x*3/5 previously.
		Likewise, if x comes from the 7s, then 3*x and 5*x will have already been seen as 7*(x*3/7) and
		7*(x*5/7). Thus we can skip pushing on to some queues in the 5s and 7s cases.
		"""
		threes = [3]
		fives = [5]
		sevens = [7]
		x = 1
		k -= 1

		while k > 0:
			if threes[0] <= fives[0] and threes[0] <= sevens[0]: # 3s has smallest
				x = threes.pop(0)
				threes.append(3*x)
				fives.append(5*x)
			elif fives[0] <= threes[0] and fives[0] <= sevens[0]: # 5s has smallest
				x = fives.pop(0)
				fives.append(5*x) # skip pushing on to 3s
			else: # 7s has smallest
				x = sevens.pop(0) # skip pushing on to 3s and 5s
			sevens.append(7*x) # always push on to 7s

			k -= 1
				
		return x

assert nine(20, brute_force=True) == nine(20, brute_force=False) == 175
assert nine(1, brute_force=True) == nine(1, brute_force=False) == 1

def ten(arr):
	"""If I didn't have to worry about O(1) space, then just use a Counter. If I had the mode, then
	checking whether it occurs >half the time is O(N) time, O(1) space, but all the solutions I can
	find for getting the mode use some kind of memory, which makes me think majority element is the
	computationally easier problem, so let's exploit the majorityness.

	What does majorityness get us? The key insight is that if x is the majority element of the whole
	array, then removing equal numbers of x and non-x from the array leaves us with x as still the
	majority in what's left. And removing elements that aren't x just makes x all the easier to find.

	Let's assume the first element, a, is the majority and start checking whether it is. As soon as we
	get equal counts of a and non-a, stop short, and toss all those elements. Begin checking whether the
	next element after this is the majority element of the remaining array. If we never reach the toss
	condition, then a is x, the majority we've been looking for.

	If a == x, but we still toss due to not finding enough x early on, then we've tossed equal numbers of
	x and non-x, and we'll still find x to be a majority later. If a != x, then it's possible we've
	paried a with all xs, and we've again spent an equal number of xs to eliminate as, but it's also
	possible we've paired with some non-xs, and more of what we remove is extraneous, leaving an even
	higher concentration of xs to be found later. In this way, we'll validate some a as our x in O(n).

	Caveat: It is possible to have a degenerate case where we remove a bunch of non-x elements early,
	thereby *raising the "concentration"* of x later, making x the majority element of a later subarray,
	but, when we "dilute" with those previously-tossed elements, not of the whole array. So do one last
	sweep to check the majority candidate works. It's not possible to have more than one candidate, so
	if this one doesnt' work, there is no majority element for the array.
	"""
	c = 0 # count of |matches| - |non matches|
	for i,a in enumerate(arr): # O(n)
		if c == 0:
			maj = a
			c = 1
		elif a == maj:
			c += 1
		else:
			c -= 1

	match = sum(maj == a for a in arr) # O(n)

	return maj if match > len(arr)//2 else -1

assert ten([1, 2, 5, 9, 5, 9, 5, 5, 5]) == 5
assert ten([1, 2, 5, 9, 5, 9, 5, 5]) == -1
assert ten([1, 2, 5, 9, 5, 9, 5, 5, 5, 7]) == -1
assert ten([3, 1, 7, 1, 1, 7, 7, 3, 7, 7, 7]) == 7

def eleven(document, word_pairs):
	"""If I only have one word-pair, then the way to go is to iterate the document with two pointers, one
	to the location of the last occurrence of word1, and one with the last occurrence of word2. As these
	get updated, check whether abs(p1 - p2) is < the previous best, and in the end you return that best
	value.

	If I have many words, then I can preprocess the document to get a map from word -> [locations] where
	that word occurs. Then if I want to compare word1 to word2, I get their two locations lists and
	essentially do a mergesort-merge-like traversal of them to figure out what the min diff is. This is
	completely analogous to the single-word-pair case, except I'm not generating these lists as I iterate
	the document itself.
	"""
	d = defaultdict(list)
	for i,word in enumerate(document.split()):
		for punct in ".,":
			word = word.replace(punct, '')
		d[word.lower()].append(i)

	dists = []
	for word1,word2 in word_pairs:
		l1 = d[word1.lower()]
		l2 = d[word2.lower()]

		a = 0
		b = 0
		m = float('inf')
		while a < len(l1) and b < len(l2):
			if l1[a] < l2[b]:
				m = min(m, l2[b] - l1[a])
				a += 1
			else:
				m = min(m, l1[a] - l2[b])
				b += 1
		# when one or the other pointer runs off the end of its list, then every subsequent possible match
		# from iterating the opposite pointer will be farther than something we've already checked, so we're
		# safe to end.
		dists.append(m)

	return dists

lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt " + \
	"ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris " + \
	"nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit " + \
	"esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in " + \
	"culpa qui officia deserunt mollit anim id est laborum."
assert eleven(lorem_ipsum, [('Lorem', 'ipsum'), ('consectetur', 'in'), ('dolor', 'ut'), ('dolore', 'in'),
	('ut', 'in')]) == [1, 35, 9, 5, 10]

from cci_02 import Node # I'm using my Node class instead of defining BiNode

def twelve(root):
	"""Idea: in-order traveral of the tree, saving nodes in a list as we go, then iterate the list, setting
	left and right pointers as we go. I got this to work, but obviously it's O(n) space, and I'd like to do
	it in O(1) space.

	A key thing to recognize is that because my in-order traversal involved appending to a list, and I was
	later just setting left pointers = the leftward thing in that list, I didn't really need the whole list
	to set left pointers; I could just keep the last thing in that list, set left pointers as I visit nodes,
	and then set the current node to be the new last node.

	Unfortunately I don't see a way to extend this to right pointers, but it's not really a problem, because
	at the end of my in-order traversal I have a singly-linked list (via the left pointers) and the last
	thing in the list (sort of the head if left is forward), so I can access all the nodes by iterating along
	this list, setting their right pointers properly as I go.
	"""
	last = [None] # gotta box it up in a list to make it accessible in the inner function
	def inorder(node): # O(log n) space on the stack
		if node:
			inorder(node.prev)
			node.prev = last[0]
			last[0] = node
			inorder(node.next)

	inorder(root)
	
	last = last[0] # unbox
	last.next = None # first right pointer goes to None
	while last.prev:
		last.prev.next = last
		last = last.prev

	return last
	# Gayle's solution to this is purely recursive and works on a principle of returning the start and
	# end of a doubly-linked list for left and right portions. The run time and space complexity aren't
	# really better. I prefer my solution simply because it looks like every other in-order solution I've
	# had to write.

#		  5
#	  11     20
#	8      18   2
#	  3
# To define a tree with my Node class, prev is sort of like left, and next is sort of like right
tree = Node(8, prev=Node(5, prev=Node(2, next=Node(3))), next=Node(18, prev=Node(11), next=Node(20)))
dll = twelve(tree)
assert str(dll) == "2<->3<->5<->8<->11<->18<->20->"
assert dll.prev is None

def thirteen(nospace, dictionary):
	"""I had to code this one from the solutions before I could begin to understand it. We're splitting
	problem(string) into prefix + " " + subproblem(rest of string), basically deciding where to put that
	space.

	We can evaluate the cost of putting a space at any given location = len(prefix) if the prefix isn't
	in the dictionary or 0 if the prefix is a valid word + cost of the subproblem. We're looking for
	the optimal cost.

	We also need the string parsing associated to the optimal cost, so our recursive function needs to
	return both the cost and the parsing.

	The problem/subproblem is defined based on where the "rest of string" after the space begins. This
	can be uniquely identified by just a begin index, and for the total problem, we'll just begin at 0.

	The base case is when we run off the end of the string, in which case the cost is 0 and the parsing
	is the empty string.

	The recurrence for cost becomes T[i] = min over j in [i, n) { cost(string[i:j+1]) + T[j+1] }
	"""
	memo = {} # memoize, otherwise this takes stupid long on repeated subproblems
	def recurse(i):
		if i in memo: return memo[i]
		if i == len(nospace): return (0, "") # we've run off the end (base case)

		best = float('inf') # best cost found for this i
		parsing = "" # the string parsing associated with that best cost
		prefix = "" # the pre-space part of the string, for which we can take raw cost directly

		j = i
		while j < len(nospace):
			prefix += nospace[j]
			n_invalid = 0 if prefix in dictionary else len(prefix) + 1 # + 1 term to penalize extra
				# spaces, since four prefixes lined up as j e s s is traversed before the single
				# prefix jess, and otherwise both have the same invalid count.

			if n_invalid < best: # if that alone gives worse cost, then no need to recurse
				n_invalid_sub, parsing_sub = recurse(j+1)

				subproblem_cost = n_invalid + n_invalid_sub
				if subproblem_cost < best: # if we're still better, then adopt as new best
					best = subproblem_cost
					parsing = prefix
					if len(parsing_sub) > 0: parsing += ' ' + parsing_sub

					if best == 0: break # then we've found an optimal division, so stop
			j += 1

		memo[i] = (best, parsing)
		return best, parsing

	return recurse(0)[1]

dictionary = {"look", "looked", "looks", "just", "like", "her", "brother", "time", "this", "is",
	"favorite", "food"}
assert thirteen("jesslookedjustliketimherbrother", dictionary) == "jess looked just like tim her brother"
assert thirteen("thisismikesfavoritefood", dictionary) == "this is mikes favorite food"

import heapq

def fourteen(arr, k, use_heap=False): # smallest k
	if use_heap:
		"""The canonical way to do this one is with a heap in O(n log k)
		"""
		heap = [-x for x in arr[:k]] # min heap, and I'll need to check against the *largest* of the min values
		heapq.heapify(heap)			# so put -val in the heap

		for a in arr[k:]:
			if -a > heap[0]: # if -a > top of heap, then |a| is smaller than |top of heap|
				heapq.heappush(heap, -a)
				heapq.heappop(heap)

		return [-x for x in heap]
	else:
		"""If k is large, and you're willing to rely on a nondeterministic algorithm, then there is an E[O(n)]
		algorithm: rank selection!

		1.	Pick a random pivot in the array, and partition elements around the pivot s.t. <=pivot end up to
			the left, keeping track of the quantity that end up to the left, and the quantity that have value
			equal to the pivot.
		2.	- If k falls in [|left|, |left U pivot|), then the k things <= to the pivot are properly placed
			to the left, and we're done.
			- If k falls in [0, |left|), then we've separated somewhat, but the k we want to find are on the
			left still mixed with a few others. Recurse on just the left side.
			- If k falls in [|left U pivot|, |all|), then the left and pivot sets are properly to the left,
			but we need an additional k - |left U pivot| elements from the right to get to k. Recurse on just
			the right side, now looking for the k - |left U pivot|th thing on that side.
		
		In expectation, you'll divide the array in two each time you choose a pivot, but then you recurse on
		only one side and do O(n) work in each subproblem, so the master theorem-style recurrence looks like:
		T(n) = T(n/2) + O(n) -> Theta(n) https://www.nayuki.io/page/master-theorem-solver-javascript
		"""
		def rank(l, r, i):
			"""Order arr[l:r+1] so that the i smallest things come first.
			"""
			pivot = arr[randint(l,r)]
			lsz, psz = partition(l, r, pivot)

			if i < lsz: # k is to the left
				rank(l, l + lsz - 1, i)
			elif i < lsz + psz: # k lands in the pivot set
				return
			else: # k is to the right
				rank(l + lsz + psz, r, i - lsz - psz)

		def partition(l, r, pivot):
			"""Given a portion of an array and a pivot, swap elements in the array so everything <=pivot
			lies to left and >pivot lies to right, and return the sizes of the left and pivot sets.
			"""
			l0 = l
			m = l # middle, ends up at the right edge of left U pivot, but throughout it's our working index
			while m <= r:
				if arr[m] < pivot: # middle is smaller than the pivot, and left can have any value,
					t = arr[l]	# so swap arr[m] with arr[l] so we *know* the thing at left is <pivot
					arr[l] = arr[m]
					arr[m] = t
					l += 1 # we know what's at l is gucci, so consider next l
					m += 1 # don't let middle fall behind left pointer

				elif arr[m] == pivot: # no need to swap anything, just move middle pointer
					m += 1

				elif arr[m] > pivot: # middle is larger than the pivot, and right has unknown value, so
					t = arr[r]	# swap arr[m] with arr[r] so we *know* the thing at the right is >pivot
					arr[r] = arr[m]
					arr[m] = t # wildcard, bitches!
					r -= 1 # we know what's at r is gucci, so consider next r

			return l - l0, m - l

		rank(0, len(arr)-1, k)
		return arr[:k]

arr = list(range(100)) + list(range(1, 4))
for i in range(3):
	two(arr) # I could import shuffle, or I could use the one I've defined here.
	assert sorted(fourteen(arr, 10, use_heap=True)) == sorted(fourteen(arr, 10, use_heap=False)) == \
		[0, 1, 1, 2, 2, 3, 3, 4, 5, 6]

def fifteen(words):
	"""I'm taking inspiration from 13 here. Basically we can split a word in to a prefix and remainder.
	If the prefix is a valid word from the dictionary, then we're in business, and we just have to
	check that the remainder is also composed of words from the dictionary. The base cases are when the
	remainder is a valid word (positive), or when the remainder runs off the end and becomes empty string
	(negative, so that words can't make themselves positive via prefix=word & remainder="").
	"""
	memo = {} # I'm finding it difficult to get a case where this is triggered, but in theory it helps.
	def recurse(word, i): # boolean, whether composed of other words
		if word[i:] in memo: return memo[word[i:]]

		if i > 0 and word[i:] in words: return True
		if i == len(word): return False # base case: run off the end -> True

		prefix = ""
		j = i
		while j < len(word): # split dogwalker into d ogwalker, then do gwalker, etc.
			prefix += word[j]
			if prefix in words and recurse(word, j+1):
				if word[i:] not in words: memo[word[i:]] = True
				return True
			j += 1

		if word[i:] not in words: memo[word[i:]] = False
		return False

	longest = 0
	best = ""
	for word in words:
		if len(word) > longest and recurse(word, 0):
			longest = len(word)
			best = word
	return best

words = {"cat", "banana", "ba", "dog", "nana", "walk", "walker", "dogwalker", "beast"}
assert fifteen(words) == "dogwalker"
words.add("catbanananana")
assert fifteen(words) == "catbanananana"

def sixteen(minutes, recurse=False):
	"""The masseuse has a sequence 30 15 60 75 45 15 15 45. With the first appointement she can do one
	of two things: accept or reject. Say she accepts, then we know she can't take the next appointment,
	and the problem becomes [30] 15 (60 75 45 15 15 45). That is, she banks the 30, and to it she can
	add the best solution to the subproblem in parentheses. Say she rejects, then the best she can do
	is 30 (15 60 75 45 15 15 45), that is the best solution to the subproblem in parentheses, which,
	unlike last case, includes the neighboring 15. The base cases are when we get down to two or fewer
	appointments, where the better choice is to pick the max, or the one.
	"""
	n = len(minutes)

	if recurse:
		memo = {} # This is O(n) with memoization, but it's hard to see,
		def recurse(i): # and the recursion and memoization means you're using O(n) extra space
			if i in memo: return memo[i]

			if i == n-1: return minutes[-1]
			elif i == n-2: return max(minutes[-2:])

			r = max(minutes[i] + recurse(i+2), recurse(i+1))
			memo[i] = r
			return r

		return recurse(0)
	else:
		"""This really can be done iteratively. T[i] = max(minutes[i] + T[i+2], T[i+1]). Because this
		recurrence doesn't depend on anything long-term, I can keep all I need in a few variables.
		"""
		plus_two = minutes[-1] # start at end of table, where we have base cases
		plus_one = max(minutes[-2:])
		for t in reversed(minutes[:-2]): # obviously O(n) time and O(1) space
			now = max(t + plus_two, plus_one)
			plus_two = plus_one # back up these pointers
			plus_one = now

		return now

minutes = [30, 15, 60, 75, 45, 15, 15, 45]
assert sixteen(minutes, recurse=True) == sixteen(minutes, recurse=False) == 180
minutes = [75, 105, 120, 75, 90, 135]
assert sixteen(minutes, recurse=True) == sixteen(minutes, recurse=False) == 330

def seventeen(b, T):
	"""I thought of Rabin-Karp for this, which would be O(b |T|)ish, I think. The hints seem to want you
	to use a prefix tree, which makes it O(bt), where t is the length of the longest string in T (though
	you'll dip out a lot earlier most times in practice). Which to prefer kind of depends on whether you
	have few long strings or many short ones to match against. The problem says "small", though.
	"""
	n = len(b)

	prefix_tree = {} # My favorite way to do prefix trees in python is just telescoping dictionaries.
	for t in T: # O(|T| * t)
		d = prefix_tree
		for c in t:
			if c not in d:
				d[c] = {}
			d = d[c]
		d[0] = t # to denote the end of something, handy to keep the string in the termination node
	
	mapping = defaultdict(list) # from t in T -> lists of locations 

	for i in range(n): # O(b)
		d = prefix_tree

		j = i
		while j < n and d: # O(t)
			c = b[j]
			d = d.get(c, None)
			if d and 0 in d: # then we've found a termination, so add i to the corresponding string's mapping
				mapping[d[0]].append(i)
			j += 1

	return mapping

b = "dogcatcatdogdoggocatticusfeldsparcatcatheter"
T = ["dog", "cat", "cab", "catheter"]
mapping = seventeen(b, T)
assert len(mapping) == 3
assert mapping['dog'] == [0, 9, 12]
assert mapping['cat'] == [3, 6, 17, 33, 36]
assert mapping['catheter'] == [36]

def eighteen(arr, els, counter_strategy=True):
	n = len(arr)
	m = len(els)

	if not counter_strategy:
		"""Looking at this I decided to just copy out all the locations of each element in to lists and ponder
		them. What jumped out at me is that it's very similar to finding the diff between two elements (problem
		17.11) once you've got those lists, which in turn is very similar to a mergesort merge, except here you
		could have a 3-or-more-way merge happening. But the procedure is easy: a points to the first location of
		the first el, b to the first location of the second, etc. The max difference when you put them on a
		number line is the width of the subarray containing them all, and then you scoot the lowest one forward
		and try again, keeping track of the min diff and the endpoints as you go. A neat thing to realize here
		is that because we only use one value from those notional lists at a time, we don't really need to
		precompute them; we can just iterate them along the array looking for the next occurrence of an element,
		thereby saving memory.
		"""
		els = list(els)

		pointers = [0]*len(els) # initialize pointers to point to the first of each occurrence of the elements
		for i,el in enumerate(els):
			for j,a in enumerate(arr):
				if a == el:
					pointers[i] = j
					break

		best = [0, len(arr)-1]

		while True: # O(nm), I think. Pointers take turns being the lowest one, and we have to scoot them all near
			minest = min(pointers) # the end of the array in the worst case, and we do extra work to find the min.
			maxest = max(pointers) # several O(m) things going on here. This could be a little faster with a heap
			for i,p in enumerate(pointers):
				if p == minest:
					break # leaves i at the index of the lowest pointer

			if maxest - minest < best[1] - best[0]:
				best = [minest, maxest]

			for j in range(pointers[i]+1,n): # O(n) by the time we're through all the consecutive while loops
				if arr[j] == els[i]:
					pointers[i] = j
					break # don't hit the else condition
			else: # if we didn't find a next els[i] to point to, then we've covered all the subarrays containing
				break # them all. Stop the while loop.

		return best
	else:
		"""There is a fundamentally better way based on iterating endpoints forward and keeping track of how
		many of each element lie in that range. First run the front forward until we've got at least one of
		each el, then iterate the back one forward, updating counts until we hit 0 for one of them, then
		iterate forward again until we fill that zero back in. Keep going until we've covered the array.
		"""
		rear = 0 # I'm considering rear to be inclusive and lead to be exclusive, so the range
		lead = 0 # is [rear, lead). It makes some of the loop logic nicer.
		n_nonzero = 0 # This is the trick, because if n_nonzero == m, then we don't have to iterate dictionary 
		c = defaultdict(int) # keys to tell that c has >0 at every location
		
		best = [0, n-1]
		while lead < n: # O(n)
			if n_nonzero == m: # with this rear and lead, all counts are >0
				if lead-1 - rear < best[1] - best[0]: # update best if current range is smaller
					best = [rear, lead-1]

				x = arr[rear] # we're removing an x from our range
				if x in els: # if it's an element we care about, then update the count
					c[x] -= 1
					if c[x] == 0: # if the count hits zero, then unset the flag
						n_nonzero -= 1
				rear += 1

			else: # not all counts are >0
				x = arr[lead] # we're adding this element to our range
				if x in els: # if it's one we care about
					if c[x] == 0: # then if it's the one we're missing, we can set the flag
						n_nonzero += 1
					c[x] += 1 # and update the count
				lead += 1

		return best

arr = [7, 5, 9, 0, 2, 1, 3, 5, 7, 9, 1, 1, 5, 8, 8, 9, 7]
els = set([1, 5, 9])
assert eighteen(arr, els, counter_strategy=False) == eighteen(arr, els, counter_strategy=True) == [7,10]

def nineteen(arr):
	"""If we're missing one number, then this problem has a famous solution: len(arr) = N - 1 -> sum from
	1 to N is N(N+1)//2 = k. sum(arr) = k'. missing = k - k'. O(n) time, O(1) space, very clean.

	If we're missing a second number, then we can do something similar: len(arr) = N - 2 -> sum from 1 to
	N is N(N+1)//2 = k. sum(arr) = k'. missing1 + missing2 = k - k'. The problem is we have two unknowns,
	so we need two equations to lock them down.

	I initially tried xoring everything, but mixing numerical and boolean algebra resulted in nonunique
	solutions: xor([1,2,3,5,6,8,9]) = 2, xor([1..9]) = 1 -> xor = 3. sum diff comes out to 11, so
	m1^m2 = 3, m1+m2=11. The pair (4,7) solves it, but so does (5,6)!

	A good second equation candidate is the sum of the squares of the numbers, but there are many choices.
	Upon searching, I find it also has a closed form: N(N + 1)(2N + 1)//6. Very cool.
	https://www.quora.com/What-is-the-derivation-for-the-sum-of-square-of-1st-n-natural-numbers
	"""
	N = len(arr) + 2
	k = N*(N+1)//2
	k2 = N*(N+1)*(2*N+1)//6
	kprime = 0
	k2prime = 0
	for x in arr:
		kprime += x
		k2prime += x**2
	A = k - kprime
	B = k2 - k2prime

	# m1 + m2 = A -> m2 = A - m1
	# m1^2 + m2^2 = B -> m1^2 + (A-m1)^2 = B -> 2m1^2 - 2Am1 + (A^2 - B) = 0
	m1 = int(2*A - (4*A**2 - 4*2*(A**2-B))**0.5)//4
	m2 = A - m1
	return m1, m2

arr = [1,2,3,5,6,8,9]
assert sorted(nineteen(arr)) == [4,7]

class Twenty:
	def __init__(self):
		self.top_half = [] # min heap
		self.bottom_half = [] # max heap

	def push(self, a):
		"""Let's say I want the lower heap to always be equal size to the top heap, or +1 larger. (You
		could also allow 1 smaller, but it adds more cases to the logic.) Along comes a.
		If |bottom| == |top|
			then if a > top of the top heap, then it has to go in the top heap, and then we have to move
				the top of the top heap to the bottom heap to make the bottom heap the larger.
			if a <= top of the top heap, then a can just go in the bottom heap, and no rebalance
		If |bottom| > |top|
			then if a < -top of the bottom heap, then it (as -a) needs to go in the bottom heap, and we
				neet to rebalance by moving the top of the bottom heap to the top heap
			if a >= -top of the bottom heap, then a can safely be put in the top heap, and no rebalance
		"""
		if len(self.top_half) == len(self.bottom_half):
			if len(self.top_half) > 0 and a > self.top_half[0]: # len >0 check to handle very beginning when
				heapq.heappush(self.top_half, a)				# both heaps are empty
				heapq.heappush(self.bottom_half, -heapq.heappop(self.top_half))
			else:
				heapq.heappush(self.bottom_half, -a)
		else:
			if a < -self.bottom_half[0]:
				heapq.heappush(self.bottom_half, -a)
				heapq.heappush(self.top_half, -heapq.heappop(self.bottom_half))
			else:
				heapq.heappush(self.top_half, a)

	def median(self):
		"""If there are an even number of things, the median is defined as the average of the two center
		ones. Otherwise, median is the actual center thing.
		"""
		if len(self.top_half) == len(self.bottom_half):
			return (self.top_half[0] - self.bottom_half[0])/2.0
		else: # len(self.bottom_half) > len(self.top_half)
			return -self.bottom_half[0]

medi = Twenty()
arr = list(range(-1000, 1000))
two(arr) # shuffle
for i,a in enumerate(arr):
	medi.push(a)
	assert medi.median() == numpy.median(arr[:i+1])

def twentyone(arr, simplified=True):
	if not simplified:
		"""The idea here is that the tallest bar divides the problem in to left and right, and then the
		next-tallest bar to each side will hold in the most water, and then we search for the next-tallest
		again to each side of those until we run out, subtracting away submerged bars by iterating between
		tallest and next-tallest as we go.

		To support finding the next-tallest to each side quickly, I create a couple dictionaries to map from
		bar height -> location and height of next-tallest.
		"""
		l = {} # grow to O(n) space
		r = {}

		m = 0
		x = 0
		for i,a in enumerate(arr): # O(n)
			if a > m:
				l[a] = (m, x) # keep track of where next-tallest to left was
				m = a
				x = i

		m = 0
		x = len(arr) - 1
		for i in range(len(arr)-1, -1, -1): # O(n)
			a = arr[i]
			if a > m:
				r[a] = (m, x) # keep track of where next-tallest to right was
				m = a
				x = i

		# m = 6, x = 5
		s = 0
		c = m
		j = x
		while c in r: # chain through next-tallests to the right
			c, k = r[c]
			for i in range(j+1, k): # O(n) all together
				s -= arr[i] # subtract out submerged bars
			s += c*(k - j - 1)
			j = k

		c = m
		j = x
		while c in l: # chain through next-tallests to the left
			c, k = l[c]
			for i in range(k+1, j): # O(n) all together
				s -= arr[i] # subtract out submerged bars
			s += c*(j - 1 - k)
			j = k

		return s
	else:
		"""The above is less elegant than I'd like, because it depends on juggling a lot of variables and
		careful indexing. An alternative solution is to keep the running-maxes for left and right passes in
		arrays (or could be maps, but we'll need to key on exactly each location anyway). The water height
		at each location is now given by the min of the maxes to each location's left and right, and if
		there is a histogram bar here too, then that displaces some water.
		"""
		r = [arr[-1]]
		for a in reversed(arr[:-1]): # build running maxes for right in O(n)
			r.insert(0, max(a, r[0]))
		
		l = float('-inf') # the running max for the left, just a single variable to save space
		s = 0
		for i,a in enumerate(arr):
			l = max(a, l) 			# height for location i is min(left_max[i], right_max[i])
			s += min(l, r[i]) - arr[i] # subtract off any histogram bar height that lives here too

		return s

arr = [0,0,4,0,0,6,0,0,3,0,5,0,1,0,0]
assert twentyone(arr, simplified=False) == twentyone(arr, simplified=True) == 26
arr = [0,0,4,0,0,6,0,0,3,0,8,0,2,0,5,2,0,3,0,0]
assert twentyone(arr, simplified=False) == twentyone(arr, simplified=True) == 46

def twentytwo(source, dest, dictionary):
	"""Use a BFS. I'm choosing to search from both ends, because this is going to get explosively large for
	a large dictionary, and we can take the number of nodes we have to explore down dramatically if we search
	from both ends. E.g., say each time we explore a node we find b additional nodes to add to the queue,
	and say the distance between the source and destination is k, then we search b^k. But if we divide in two
	to search from both ends, we get there in 2*b^(k/2), and our visited sets get to stay smaller likewise.
	"""
	one_away = defaultdict(set) # O(|dictionary|^2) to generate, this basically becomes the graph
	for A in dictionary:
		for i in range(len(A)):
			k = A[:i] + '_' + A[i+1:]
			for B in dictionary:
				if A[:i] == B[:i] and A[i+1:] == B[i+1:]:
					one_away[k].add(A)
					one_away[k].add(B)
	A = source
	B = dest
	ancestorsA = {A: None} # also serves as visited set
	ancestorsB = {B: None}
	qA = []
	qB = []

	while A not in ancestorsB and B not in ancestorsA:

		for i,c in enumerate(A): # search forward
			k = A[:i] + '_' + A[i+1:]
			for candidateA in one_away[k]:
				if candidateA not in ancestorsA:
					qA.append(candidateA)
					ancestorsA[candidateA] = A

		for i,c in enumerate(B): # search backward
			k = B[:i] + '_' + B[i+1:]
			for candidateB in one_away[k]:
				if candidateB not in ancestorsB:
					qB.append(candidateB)
					ancestorsB[candidateB] = B

		A = qA.pop(0)
		B = qB.pop(0)

	# recover the path
	K = A if A in ancestorsB else B
	path = [K]
	Y = ancestorsA[K]
	while Y is not None:
		path.insert(0, Y)
		Y = ancestorsA[Y]
	Y = ancestorsB[K]
	while Y is not None:
		path.append(Y)
		Y = ancestorsB[Y]
	return path

dictionary = {'damp', 'like', 'lamp', 'limp', 'lime', 'like', 'bike', 'camp', 'time', 'line'}
assert twentytwo('damp', 'like', dictionary) == ['damp', 'lamp', 'limp', 'lime', 'like']

def twentythree(M):
	"""There is 1 NxN square in M, 4 (N-1)x(N-1) squares, 9 (N-2)x(N-2) squares, 25 (N-3)x(N-3) squares ...
	Sum this up and it's the sum of squared 1,2,3,4,5 ..., which in 17.19 I learned is N(N+1)(2N+1)/6. This
	means there are O(N^3) subsquares in M. The hints say that's the runtime we should seek to achieve, so
	it's reasonable to think this comes from fundamentally having to visit every subsquare in the worst
	case.

	To naively calculate whether a square has an all-black border, we have to iterate along the four edges,
	incurring an additional O(N) cost. It strikes me that the way to get down to O(N^3) all together is
	probably to make this border check O(1). What preprocessing can we do? The last hint kind of spells it
	out, but it took me some reflection to understand: Let's loop over the array storing how many black
	squares are above and to the left of each square. Then if we want to check whether a border is all black
	we can take the diff of these num-black values for corners of the subsquare, because if the diff == the
	side length of the subsquare, then all cells in between must be black.
	"""
	N = len(M)
	T = numpy.zeros((2, N+1, N+1), dtype=int) # I need (N+1)x(N+1) to index cleanly
	T[0,1:,1:] = numpy.cumsum(M, axis=0) # I'm storing black above in front, black left in back
	T[1,1:,1:] = numpy.cumsum(M, axis=1) # O(N^2) to build T

	# generate all subsquares
	for L in range(N, 0, -1):
		for i in range(N-L+1):
			for j in range(N-L+1):
				# the corners are at (i,j), (i+L,j), (i,j+L), (i+L,j+L) in T. This actually runs off the
				# square in M, and I need to be careful to subtract elements which are L apart in T.
				# For example: M = [[0 0 0 0] yields T[1] =[[0 0 0 0 0]
				#					[0 1 1 1]				[0 0 0 0 0]
				#					[0 1 0 1]				[0 0 1 2 3]
				#					[0 1 1 1]]				[0 0 1 1 2]
				#											[0 0 1 2 3]]
				# Say i,j = 1,1 and L = 3, then to properly check top and bottom rows, we have to subtract
				# T[4,4] - T[4,1] and T[2,4] - T[2,1]. Notice that we're operating over a rectangle that
				# runs off the square only in the horizontal direction. For vertical edges, it's analogous.
				if T[0,i+L,j+L] - T[0,i,j+L] == L and T[0,i+L,j+1] - T[0,i,j+1] == L and \
					T[1,i+L,j+L] - T[1,i+L,j] == L and T[1,i+1,j+L] - T[1,i+1,j] == L:
					return i,j,L

M = [[0, 0, 0, 0,],
	[0, 1, 1, 1,],
	[0, 1, 0, 1,],
	[0, 1, 1, 1,]]
assert twentythree(M) == (1,1,3)
M = [[0, 1, 0],
	[1, 1, 0],
	[1, 1, 0]]
assert twentythree(M) == (1,0,2)

def twentyfour(M):
	"""In order to select a submatrix, we place 4 pointers: xlo, xhi, ylo, yhi. That takes us up to O(N^4).
	Then we have to sum over each submatrix, which incurs an extra O(N^2). Gross.

	First, observe that if we take a cumsum in one dimension and then a cumsum of the cumsum in the other
	dimension, we get the sum of everything up and left in the array. Then we can subtract away different
	cumsum terms to get the sum of any submatrix in O(1). That solves one problem.

	Next, consider the way we solve the 1D case: iterate along keeping a cumsum and a best, resetting the
	cumsum to 0 every time it dips below 0. If I choose the two pointers along one edge to define which rows
	I want my submatrix to be between, then I can use a very similar principle to find the best submatrix in
	O(N) by iterating along, adding the right pieces of columns, resetting as necessary. It will still take
	O(N^2) to choose where to put those two pointers, making the total run time O(N^3). We actually don't
	need the double-cumsum here either, just the cumsum of columns, which makes indexing easier.
	"""
	N = len(M)
	T = numpy.zeros((N+1, N), dtype=int)
	T[1:,:] = numpy.cumsum(M, axis=0)

	best = float('-inf')
	rect = (0, float('inf'), 0, float('inf'))
	for r1 in range(N): # choose two row pointers (inclusive)
		for r2 in range(r1, N):
			s = 0
			c1 = 0
			for c2 in range(N): # iterate rightward, including new columns
				s += T[r2+1,c2] - T[r1,c2] # get the sum of the fractional column in O(1)
				if s > best or s == best and (r2-r1)*(c2-c1) < (rect[1]-rect[0])*(rect[3]-rect[2]):
					best = s # if we get a higher sum, or the same sum in smaller area, save
					rect = (r1,r2,c1,c2)
				elif s <= 0: # if we dip to or below zero, bring the left edge of the submatrix forward
					s = 0
					c1 = c2+1 # +1 because we don't want to include this column that just made us go negative

	return best, rect

assert twentyfour(M) == (5, (0, 2, 0, 1))
M = [[0, 0, 0, 0,],
	[0, 1, 1, 1,],
	[0, 1, 0, 1,],
	[0, 1, 1, 1,]]
assert twentyfour(M) == (8, (1, 3, 1, 3))
M = [[randint(-10,10) for x in range(10)] for y in range(10)]
# I'm actually coding the O(N^6) brute-force way to check this
best = 0
rect = (0, float('inf'), 0, float('inf'))
for r1 in range(10):
	for r2 in range(r1,10):
		for c1 in range(10):
			for c2 in range(c1,10):
				s = 0
				for i in range(r1,r2+1):
					for j in range(c1,c2+1):
						s += M[i][j]
				if s > best or s == best and (r2-r1)*(c2-c1) < (rect[1]-rect[0])*(rect[3]-rect[2]):
					best = s
					rect = (r1,r2,c1,c2)
assert twentyfour(M) == (best, rect), [twentyfour(M), (best, rect)]

def twentyfive(dictionary):
	"""This problem is legit insane. It has recursion, prefix trees, hash maps, clever looping, and object
	oriented design. This is just my best translation of the solution into Python. Gayle stresses that for
	something this complex, you're going to be writing pseudocode, but I'd have a lot of trouble coming up
	with a workable solution design without some guidance from the interviewer. Guidance is the only way to
	make this one not evil.
	"""
	by_length = defaultdict(list)
	longest = 0
	for x in dictionary:
		by_length[len(x)].append(x)
		longest = max(longest, len(x))

	prefix_trees = {} # map from word length -> prefix tree

	class Rectangle:
		"""This I coded myself just from the general idea of what the methods need to do. Keeping
		some of this complexity inside makes the makeRect recursion tractable."""
		def __init__(self, data=[]):
			self.rows = data

		def __len__(self):
			return len(self.rows)

		def __repr__(self):
			return '\n'.join(self.rows)

		def isValid(self, vocab):
			"""Check whether all column-words are in by_length[j]"""
			for k in range(len(self.rows[0])):
				col = ""
				for j in range(len(self.rows)):
					col += self.rows[j][k]

				if col not in vocab: return False

			return True

		def isPartialValid(self, prefix_tree):
			"""Check whether all column-words are prefixes in prefix_trees[j]"""
			if len(self.rows) == 0: return True

			for k in range(len(self.rows[0])):
				d = prefix_tree
				for j in range(len(self.rows)):
					c = self.rows[j][k]
					if c in d:
						d = d[c]
					else:
						return False

			return True

		def stack(self, word):
			"""Return a new rectangle with an additional row-word"""
			return Rectangle(self.rows + [word])

	def makeRect(i, j):
		if len(by_length[i]) == 0 or len(by_length[j]) == 0: return

		# Make prefix tree for words of length j. i dimension doesn't need a prefix tree.
		# Doing this here means we only make the prefix trees we need.
		if j not in prefix_trees:
			prefix_trees[j] = {}
			for word in by_length[j]: # O(|dictionary| * |word|)
				d = prefix_trees[j]
				for c in word:
					if c not in d:
						d[c] = {}
					d = d[c]
				d[0] = word # at the leaf, store the word

		return makePartialRect(i, j, Rectangle())

	def makePartialRect(i, j, rect):
		if len(rect) == j and rect.isValid(by_length[j]):
			return rect # check if all columns are valid words

		elif rect.isPartialValid(prefix_trees[j]):
			for next_word in by_length[i]:
				r = makePartialRect(i, j, rect.stack(next_word))
				if r: return r

	# This loop iterates from largest possible rectangle down to smallest
	for z in range(longest**2, 0, -1):
		for i in range(1, longest+1): # z = i * j
			if z % i == 0:
				j = z//i
				if j <= longest:
					# attempt to create an i x j rectangle
					r = makeRect(i, j)
					if r: return r

dictionary = [x.strip() for x in open('dictionary.txt') if len(x) <= 6] # limit size so this doesn't take ages
assert str(twentyfive(dictionary)) == """aahed
abaca
haars
ecrus
dassy"""

def twentysix(documents, sol=2):
	similarities = {}
	
	if sol == 1:
		"""Brute force is just compare all documents against all other documents, calculating IoU for each
		pair, storing the nonzero ones.
		"""
		documents = [set(d) for d in documents]

		for i,A in enumerate(documents): # O(|documents|^2 * |length of document pairs|)
			for j in range(i+1,len(documents)):
				B = documents[j]
				I = len(A & B)
				iou = I / (len(A) + len(B) - I)
				if iou > 0:
					similarities[(i,j)] = iou

		return similarities

	elif sol == 2:
		"""A quicker solution is to go element-wise, mapping from element to which documents contain that
		element.
		"""
		els = defaultdict(list) # keep track of where words appear
		for i,d in enumerate(documents): # build in O(|length of all documents|)
			for w in d:
				els[w].append(i)

		I = defaultdict(int) # count up number of commonalities between document pairs
		for w in els:
			l = els[w] # all possible (i,j) pairs in l get +1 to the size of their intersection
			m = len(l)
			if m > 0:
				for a,i in enumerate(l):
					for b in range(a+1, m):
						j = l[b]
						I[(i,j)] += 1 # 

		for i,j in I: # calculate IoUs from nonzero intersections
			similarities[(i,j)] = I[(i,j)] / (len(documents[i]) + len(documents[j]) - I[(i,j)])

		return similarities

documents = [[14,15,100,9,3], [32,1,9,3,5], [15,29,2,6,8,7], [7,10]]
assert twentysix(documents, sol=1) == twentysix(documents, sol=2) == {(0, 1): 0.25, (0, 2): 0.1, (2, 3): 1./7}
