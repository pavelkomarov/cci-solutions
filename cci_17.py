
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











