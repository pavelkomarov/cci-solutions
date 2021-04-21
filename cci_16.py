
def one(ab, xor=False):
	if not xor: # do it the diff way
		ab[0] = ab[1] - ab[0] # a = b - a
		ab[1] = ab[1] - ab[0] # b = b - (b-a) = a
		ab[0] = ab[0] + ab[1] # a = (b-a) + a = b
	else: # do it the xor way
		ab[1] = ab[0] ^ ab[1] # b = a ^ b
		ab[0] = ab[0] ^ ab[1] # a = a ^ (a ^ b) = b
		ab[1] = ab[0] ^ ab[1] # b = b ^ (a ^ b) = a

for ab in [[5, 7], [-13, 2]]:
	for xor in [False, True]:
		before = [x for x in ab]
		one(ab)
		assert ab == list(reversed(before))

from collections import Counter

def two(word, book=None, vocab_freq=None):
	if book is None and vocab_freq is None:
		raise ValueError("I need the book or a vocab_freq map")

	if vocab_freq:
		return vocab_freq[word], None
	else:
		vocab_freq = Counter(book.split())
		return vocab_freq[word], vocab_freq

book = "a quick brown fox jumped over the lazy dog dog doggo yo"
c, d = two("dog", book=book)
assert c == 2
c, _ = two("jumped", vocab_freq=d)
assert c == 1
try:
	two("brown")
	assert False
except ValueError:
	assert True

def three(line1, line2):
	v1 = line1[1][0] == line1[0][0] # whether the lines are vertical (have infinite slope),
	v2 = line2[1][0] == line2[0][0] # which breaks the math
	if v1 and v2: # parallel lines
		if line1[0][0] == line2[0][0]: # on same infinite line
			u1 = max(line1[0][1], line1[1][1])
			l1 = min(line1[0][1], line1[1][1])
			u2 = max(line2[0][1], line2[1][1])
			l2 = min(line2[0][1], line2[1][1])
			if u1 == l2: return line1[0][0], u1 # the lines just barely touch
			elif l1 == u2: return line1[0][0], u2
			elif u1 <= u2 and u1 > l2: return True # the lines overlap
			elif u2 <= u1 and u2 > l1: return True
			else: return False # The segments are separated
		else: return False # no chance of intersection

	if not v1:
		m1 = float(line1[1][1] - line1[0][1])/(line1[1][0] - line1[0][0])
		b1 = line1[0][1] - m1*line1[0][0]
	if not v2:
		m2 = float(line2[1][1] - line2[0][1])/(line2[1][0] - line2[0][0])
		b2 = line2[0][1] - m2*line2[0][0]

	if not v1 and not v2:
		if abs(m1 - m2) < 1e-9: # parallel segments
			if abs(b1 - b2) < 1e-9: # on same infinite line
				u1 = max(line1[0][0], line1[1][0])
				l1 = min(line1[0][0], line1[1][0])
				u2 = max(line2[0][0], line2[1][0])
				l2 = min(line2[0][0], line2[1][0])
				if u1 == l2: xi = u1 # the lines just barely touch
				elif l1 == u2: xi = u2
				elif u1 <= u2 and u1 > l2: return True # the lines overlap
				elif u2 <= u1 and u2 > l1: return True
				else: return False # The segments are separated
			else: return False # no chance of intersection
		else:
			xi = float(b2 - b1)/(m1 - m2)
	elif v1:
		xi = float(line1[0][0])
	else: # v2
		xi = float(line2[0][0])
		
	if v1: yi = m2*xi + b2
	else: yi = m1*xi + b1

	# check the intersection is on both line segments. You actually only need to check
	# either xi or yi, because the other is constrained by the equation to also lie on
	# the line segment, but I didn't see this shortcut at first.
	if (line1[0][0] <= xi <= line1[1][0] or line1[0][0] >= xi >= line1[1][0]) and \
		(line2[0][0] <= xi <= line2[1][0] or line2[0][0] >= xi >= line2[1][0]) and \
		(line1[0][1] <= yi <= line1[1][1] or line1[0][1] >= yi >= line1[1][1]) and \
		(line2[0][1] <= yi <= line2[1][1] or line2[0][1] >= yi >= line2[1][1]):
		return xi, yi
	else: return False

line1 = ((0,0), (5,5))
line2 = ((3,0), (0,3))
assert three(line1, line2) == (1.5,1.5) # normal case
line1 = ((2,2),(5,5))
assert not three(line1, line2) # non-intersecting case
line1 = ((4,0), (0,4))
assert not three(line1, line2) # parallel case
line1 = ((3,0), (0,3))
assert three(line1, line2) == True # same line case
line1 = ((2,0), (2,2))
assert three(line1, line2) == (2.0, 1.0) # vertical line case
line2 = ((3,0), (3,3))
assert not three(line1, line2) # both vertical lines case
line1 = ((1,0), (4,3))
assert three(line1, line2) == (3.0, 2.0) # other vertical line case
line2 = ((4,3), (5,4))
assert three(line1, line2) == (4.0, 3.0) # the two are parallel and touch, tip to tail
line2 = ((3,2),(5,4))
assert three(line1, line2) == True # parallel and overlap
line2 = ((5,4),(6,5))
assert not three(line1, line2) # on same line segment and not touching
# I could keep going. This problem has a ton of corner cases.

def four(board):
	d = [0]*4 # x diag1, o diag1, x diag2, o diag2
	for i in range(3):
		c = [0]*4 # x horiz, o horiz, x vert, o vert
		for j in range(3):
			if board[i][j] == 'X': c[0] += 1
			elif board[i][j] == 'O': c[1] += 1
			if board[j][i] == 'X': c[2] += 1
			elif board[j][i] == 'O': c[3] += 1
		if c[0] == 3 or c[2] == 3: return 'X'
		elif c[1] == 3 or c[3] == 3: return 'O'

		if board[i][i] == 'X': d[0] += 1
		elif board[i][i] == 'O': d[1] += 1
		if board[2-i][i] == 'X': d[2] += 1
		elif board[2-i][i] == 'O': d[3] += 1
	if d[0] == 3 or d[2] == 3: return 'X'
	elif d[1] == 3 or d[3] == 3: return 'O'

board = [['X', '', 'O'],
		 ['X', '', 'O'],
		 ['X', '', '']]
assert four(board) == 'X'
board = [['X', '', 'O'],
		 ['', 'X', 'O'],
		 ['X', '', '']]
assert four(board) is None
board = [['X', '', 'O'],
		 ['', 'X', 'O'],
		 ['', '', 'X']]
assert four(board) == 'X'
board = [['X', '', 'O'],
		 ['', 'O', 'X'],
		 ['O', '', 'X']]
assert four(board) == 'O'
# The hints ask to consider how to handle if calling multiple times. If I'm calling on the
# same board over and over, then I'd keep sums of Xs and Os along rows, cols, and diags,
# adding as moves get made. Hopefully I get told where a move was made, so I can go direct
# to updating the right row, col, and diag. If any sum hits 3, then return who it belongs to.

def five(n):
	s = 0
	d = 5 # We only need |factors of 5 <= n| = n//5, because 5 and 2 are the primes of 10 
	while True:		# and there will always be more factors of 2 than 5 to pair with
		c = n//d	# whichever factors of 5 we find. The wrinkle is that if n is also
		if c == 0: break	# divisible by a higher power of 5, then we also have to account
		s += c				# for the extra fives, so add inn//25, n//125, etc until the
		d *= 5				# division stops adding anything.

	return s

from math import factorial

for i in [4, 10, 20, 30, 1000]:
	nz = five(i)
	f = str(factorial(i))
	if nz > 0: assert f[-nz:] == '0'*nz
	assert f[-nz-1] != '0'

def six(A, B, sort_both=False):
	n = len(A)
	m = len(B)
	d = float('inf')

	# Merge approach: sort both arrays, and go through the motions of a mergesort merge
	# to find the smallest diff in what would be the merged array, which is effectively
	# what we're looking for.
	if sort_both:
		A = sorted(A)
		B = sorted(B)
		i = 0
		j = 0
		
		while i < n-1 or j < m-1:
			d = min(d, abs(A[i] - B[j]))
			if A[i] <= B[j] and i < n-1:
				i += 1
			else:
				j += 1

		return d

	# Clever alternative from the solutions, for if one array is significantly longer
	# than the other: Only sort the shorter one, and then iterate the unsorted larger
	# one, binary searching the smaller for the best pairing for each.
	else:
		S, L = (sorted(A), B) if n < m else (sorted(B), A)

		for i in L:

			lo = 0
			hi = len(S) - 1
			while lo <= hi:
				mid = (lo + hi)//2
				e = S[mid]
				if e == i: return 0

				if e < i:
					lo = mid + 1
				else:
					hi = mid - 1

			# lo is now where i would be inserted in S (could be running off the end),
			# and hi points to the thing with value just below.
			if hi != -1: # Wrinkle: if nothing in S < i, then hi = -1
				d = min(d, i - S[hi])

		return d

A = [1,3,15,11,2]
B = [23,127,235,19,8]
for sort_both in [True, False]:
	assert six(A, B, sort_both) == 3
	B.append(15)
	assert six(A, B, sort_both) == 0
	B = B[:-1]

def seven(a, b): # find max without >, <, or an if statement
	# first pass, not accounting for overflow.
	# k = ((b - a) >> 31) & 1 # b - a > 0 if b > a, so sign bit is 0 -> k = 0; k = 1 if a > b

	# Accounting for overflow is interesting and takes cleverness, so here's that. The key
	# insight is we don't have to detect overflow, because it's is only possible if the signs of
	# a and b differ, in which case we just want to select and return the positive one anyway.
	# And if there's no overflow possibility, k is as it was.
	sa = (a >> 31) & 1 # 1 if these are negative, 0 if they're positive
	sb = (b >> 31) & 1
	ab_diff = sa ^ sb # If they have different signs: If sa, then b is +, so we want k = 0.
		# If sb, then a is the + one, and we want k = 1.
	k = ab_diff*(1-sa) # basically saying "k= ab_diff and not sa"
	# If they have the same sign, then we want k = k_og, what it was in my first-pass solution.
	k += (1-ab_diff)*((b - a) >> 31 & 1) # basically saying "k or= if not ab_diff and k_og"

	return a*k + b*(1-k)

assert seven(5,2) == seven(2,5) == 5
assert seven(2**31 - 2, -15) == seven(-15, 2**31 - 2) == 2**31 - 2

def eight(n):
	d = {'0':"zero", '1':"one", '2':"two", '3':"three", '4':"four", '5':"five", '6':"six",
		'7':"seven", '8':"eight", '9':"nine"}
	p = {'2':"twenty", '3':"thirty", '4':"fourty", '5':"fifty", '6':"sixty",
		'7':"seventy", '8':"eighty", '9':"ninety"}
	t = {'10':"ten", '11':"eleven", '12':"twelve", '13':'thirteen', '14':"fourteen",
		'15':"fifteen", '16':"sixteen", '17':"seventeen", '18':"eighteen", '19':"nineteen"}
	a = ['', 'thousand', 'million', 'billion', 'trillion', 'quadrillion']

	s = ['']
	n = str(n) + " " # add space so I can grab last triplet conveniently
	if n[0] == '-':
		s[0] = "negative"
		n = n[1:]
	l = len(n) - 1

	for i in range(-(-l//3)): # ceiling function
		triplet = n[-3*(i+1)-1:-3*i-1]
		hundreds = triplet[0] if len(triplet) == 3 else False
		tens = triplet[1] if hundreds else triplet[0] if len(triplet) == 2 else False
		ones = triplet[2] if hundreds else triplet[1] if tens else triplet[0]
		
		u = []
		if hundreds and hundreds != '0':
			u.append(d[hundreds])
			u.append("hundred")
		
		if tens and tens != '0':
			u.append(p[tens] if tens != '1' else t[tens+ones])
		
		if not tens and not hundreds: # then always append, even if the answer is zero
			u.append(d[ones])
		elif not tens == '1' and ones != '0': # we know we've at least got tens
			u.append(d[ones])
		
		if not (hundreds == tens == ones == '0'): # handle the special case
			u.append(a[i])

		if len(u) > 0: s.insert(1, ' '.join(u))

	return ' '.join(s).strip()

assert eight(10300582197) == "ten billion three hundred million five hundred eighty two thousand one hundred ninety seven"
assert eight(4823749018) == "four billion eight hundred twenty three million seven hundred fourty nine thousand eighteen"
assert eight(1000090) == "one million ninety"
assert eight(-8000403) == "negative eight million four hundred three"

class Nine:

	@staticmethod
	def multiply(a, b):
		if b > a: return Nine.multiply(b, a) # faster if smaller number second

		x = 0
		flip = b < 0
		if flip: b = ~b + 1
		
		for i in range(b):
			x += a
		
		if flip: x = ~x + 1
		return x

	@staticmethod
	def divide(a, b):
		# This one is annoying, because 11//2 = -11/-2 = 5, but -11//2 = 11/-2 = -6
		# Gayle's solution actually completely ignores this complexity, just letting
		# 11/-2 = -5, but I account for it.
		if a == 0: return 0
		if b == 0: raise ValueError("denominator is zero")

		flipa = a < 0
		flipb = b < 0
		if flipa: a = ~a + 1
		if flipb: b = ~b + 1

		x = 0
		c = b
		while c < a:
			c += b
			x += 1
		# x is now = the number of times |b| goes in to |a| if they don't divide evenly.
		# If they do divide evenly, then c == a. If the answer is positive, we need to
		# add this last, but not so if the answer is negative: 10//2 = 4 + 1 = 5, but
		# -10/2 = ~4 = -5. Yet 9//2 = 4 + 0, and -9//2 = ~4 = -5.

		if flipa ^ flipb: x = ~x # no +1 here, because 5,-6 are bitwise mirrors already
		elif c == a: x += 1
		return x

	@staticmethod
	def subtract(a, b):
		return a + ~b + 1

nine = Nine()
for i in range(-10,10):
	for j in range(-10, 10):
		assert nine.multiply(i,j) == i*j
		if j != 0: assert nine.divide(i,j) == i//j
		assert nine.subtract(i,j) == i - j

def ten(peeps):
	dpeeps = [0]*102 # keep an array of how population *changes* over time, like a derivative
	for p in peeps:
		dpeeps[p[0]-1900] += 1
		dpeeps[p[1]+1-1900] -= 1 # death is applied to the *next* year's count

	t = 0
	tmax = 0
	y = 0
	for i,d in enumerate(dpeeps):
		t += d
		if t > tmax:
			tmax = t
			y = i + 1900

	return y

assert ten([(1900,1950), (1927,1985), (1945,2000), (1990,2000),
			(1987,2000), (1905,1970), (1963,1985), (1957,1975)]) == 1963

def eleven(shorter, longer, K, recurse=False):
	if recurse: # O(K^2) even with the memoization, because of the way the recursive tree
		d = set()	# branches out. I had to draw it to get this intuition.
		seen = set()

		def recurse(l_so_far, K_left):
			if K_left == 1:
				d.add(l_so_far+shorter)
				d.add(l_so_far+longer)

			else:
				for choice in [(l_so_far+shorter, K_left-1), (l_so_far+longer, K_left-1)]:
					if choice not in seen:
						seen.add(choice)
						recurse(*choice)

		recurse(0, K)
		return list(d)
	else:
		# Based on the fact all solutions made of i short planks and K-i long planks will
		# have the same length, we can just loop through finding what those lengths are.
		# We don't have to care about all 2^K specific arrangements of planks.
		d = []
		for i in range(K+1): # O(K)
			d.append((K-i)*shorter + i*longer)
		return d

assert eleven(1, 3, 10, recurse=True) == eleven(1, 3, 10, recurse=False) == \
		[10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

from xml.etree import ElementTree

def twelve(root, mapping):
	# ElementTree nodes have .tag, .text, children in, and a .attrib dictionary
	def recurse(node):
		s = []
		s.append(str(mapping[node.tag]))

		for a,b in node.attrib.items():
			s.append(str(mapping[a]))
			s.append(b)
		s.append("0")

		t = node.text.strip()
		if len(t) > 0: s.append(t)

		for child in node:
			s += recurse(child)

		s.append("0")
		return s
	
	return ' '.join(recurse(root))

mapping = {"family":1, "person":2, "firstName":3, "lastName":4, "state":5}
xml = """
<family lastName="McDowell" state="CA">
	<person firstName="Gayle">Some Message</person>
</family>
"""
assert twelve(ElementTree.fromstring(xml), mapping) == "1 4 McDowell 5 CA 0 2 3 Gayle 0 Some Message 0 0"

def thirteen(square1, square2):
	# midpoints of squares
	x1bar = (square1[0][0] + square1[3][0])/2.0
	y1bar = (square1[0][1] + square1[1][1])/2.0
	x2bar = (square2[0][0] + square2[3][0])/2.0
	y2bar = (square2[0][1] + square2[1][1])/2.0

	if x1bar != x2bar: # squares aren't perfectly stacked
		m = (y1bar - y2bar)/(x1bar - x2bar)
		b = y1bar - m*x1bar
	elif y1bar == y2bar:
		m = 0
		b = y1bar
	else: # degenerate case
		m = float('inf')
		b = float('-inf')

	return m, b

# square given as lower left, upper left, upper right, lower right
square1 = [(1,1),(1,2),(2,2),(2,1)]
square2 = [(3,2),(3,3),(4,3),(4,2)]
assert thirteen(square1, square2) == (0.5, 0.75)
assert thirteen(square1, square1) == (0, 1.5)
square2 = [(0,1),(0,3),(2,3),(2,1)]
assert thirteen(square1, square2) == (-1.0, 3.0)
square2 = [(1,0),(1,1),(2,1),(2,0)]
assert thirteen(square1, square2) == (float('inf'), float('-inf'))

from collections import defaultdict

def fourteen(points):
	n = len(points)
	d = defaultdict(int)
	g = 0

	for i in range(n):
		for j in range(i+1,n):

			if points[i][0] != points[j][0]: # squares aren't perfectly stacked
				m = (points[i][1] - points[j][1])/(points[i][0] - points[j][0])
				b = points[i][1] - m*points[i][0]
			else: # degenerate case
				m = float('inf')
				b = points[i][0] # to distinguish vertical lines, I'm making b the x intercept here

			d[m,b] += 1 # hashing with floats can be dicey in some languages, but it works here
			if d[m,b] > g:
				g = d[m,b]
				r = (m,b)

	return r

points = [(0,0), (0,1), (1,1), (2,2), (3,2), (3,1), (-1,-1), (2,-1)]
assert fourteen(points) == (1.0, 0)
points += [(3,0), (3,-1), (3,-2)]
assert fourteen(points) == (float('inf'), 3) # infinite slope means b is x intercept

def fifteen(solution, guess):
	sc = Counter(solution)
	gc = Counter(guess)

	hits = 0
	for i in range(4):
		if solution[i] == guess[i]:
			hits += 1
			l = solution[i]
			sc[l] -= 1 # so these letters can never be used for pseudo-hits
			gc[l] -= 1

	pseudos = 0
	for l in gc:
		if sc[l] > 0 and gc[l] > 0:
			pseudos += 1
			sc[l] -= 1
			gc[l] -= 1

	return hits, pseudos

assert fifteen("RGBY", "GGRR") == (1,1)
assert fifteen("RGGB", "YRGB") == (2,1)
assert fifteen("RGGB", "RRGB") == (3,0)

def sixteen(A):
	m = False

	x = float('inf')
	for i in reversed(range(len(A))): # find min as we go
		a = A[i]
		if a <= x: x = a
		else: m = i # The last index where a > min found so far (min should go left of a, but isn't)

	x = float("-inf")
	for i,a in enumerate(A): # find max as we go
		if a >= x: x = a
		else: n = i # The last index where a < max found so far (max should go right of a, but isn't)

	return (m, n) if m is not False else "already sorted"

assert sixteen([1,2,4,7,10,11,7,12,6,7,16,18,19]) == (3,9)
assert sixteen([1]*11) == "already sorted"
assert sixteen(range(10)) == "already sorted"
assert sixteen(list(range(10)) + [-1]) == (0,10)
assert sixteen([1,2,3,5,4,6,7,8]) == (3,4)

def seventeen(A):
	# The idea here is to calculate running sums, resetting every time the running sum dips
	# below 0. The largest running sum is the largest subarray sum.
	s = 0 # running sum
	w = 0 # new beginning location
	m = float('-inf') # max sum found so far
	for i,n in enumerate(A):
		s += n # running sum
		if s > m: # If the running sum beats the max,
			m = s # then the max is this sum
			v = i # It ends here
			u = w # and it started at the last start point
		if s <= 0: # reset
			w = i+1 # new start point will be after the entry that caused us to go negative
			s = 0 # sum is 0 at worst by just excluding the current entry
	
	return m, A[u:v+1]

assert seventeen([2, -8, 3, -2, 4, -10]) == (5, [3, -2, 4])
assert seventeen([-1]) == (-1, [-1])
assert seventeen([-2,1,-3,4,-1,2,1,-5,4]) == (6, [4,-1,2,1])

def eighteen(value, pattern):
	# Insight: if we have "catcatgocatgo" and "aabab", then we can let k = 1 and try a = [:k] = 'c'.
	# We know there are 3 as and 2 bs, so then the length of all the as is 3, so the length of the
	# 2 bs must be (len(value) = 13) - 3 = 10, meaning each b has a length of 5. We know b first
	# occurs as the third thing in pattern, so b = "tcatg". We can then see whether stacking these
	# together according to the pattern matches the value. Repeat for different choices of k.
	n = len(value)
	c = Counter(pattern)
	n_a = c['a']
	n_b = c['b']
	first_a = pattern.find('a')
	first_b = pattern.find('b')
	K = n+1-n_b if first_a == 0 else n+1-n_a # Has to be at least n_a or n_b characters left
												# over, so len(a), len(b) never < 1
	for k in range(1,K): # O(n)
		if first_a == 0:
			a = value[:k]
			if n_b > 0:
				len_b = (n - n_a*k)//n_b
				b = value[first_b*k:first_b*k+len_b]
		else:
			b = value[:k]
			if n_a > 0:
				len_a = (n - n_b*k)//n_a
				a = value[first_a*k:first_a*k+len_a]

		i = 0
		for x in pattern: # O(n): |a|*n_a + |b|*n_b = n
			s = a if x == 'a' else b
			l = len(s)
			if value[i:i+l] == s: # match a or b against value
				i += l
			else:
				break # this k doesn't work; try the next one
		else: # if we never break, then success, all matches
			return True

	return False # No choice of k worked, so no match

assert eighteen("catcatgocatgo","a")
assert eighteen("catcatgocatgo","b")
assert eighteen("catcatgocatgo","aabab")
assert eighteen("catcatgocatgo","aab")
assert eighteen("catcatgocatgo","bba")
assert not eighteen("catcatgocatgo", "aaab")
assert not eighteen("catcatgocatgo", "baaa")
assert not eighteen("catcatgocatgo", "aaaaa")

def nineteen(plot):
	n = len(plot)
	m = len(plot[0])
	seen = set() # if you're willing to modify the input, you can mark locations visited by setting = -1
	
	def dfs(i, j):
		seen.add((i,j))
		t = 1
		for y in [i+1, i, i-1]: # These two loops are a constant O(1), because only 9 combos
			for x in [j+1, j, j-1]:
				if (y,x) not in seen and 0 <= y < n and 0 <= x < m and plot[y][x] == 0:
					t += dfs(y, x) # Goes as deep as the pond is large, which is at most O(n*m)
		return t

	ponds = []
	for i in range(n):
		for j in range(m):
			if (i,j) not in seen and plot[i][j] == 0:
				ponds.append(dfs(i, j))

	return ponds

plot = [[0, 2, 1, 0],
		[0, 1, 0, 1],
		[1, 1, 0, 1],
		[0, 1, 0, 1]]
assert set(nineteen(plot)) == set([2,4,1])

def twenty(digits, dictionary):
	digits = str(digits)
	n = len(digits)
	d = {'0':'', '1':'', '2':'abc', '3':'def', '4':'ghi', '5':'jkl',
		'6':'mno', '7':'pqrs', '8':'tuv', '9':'wxyz'}
	
	def traverse(prefix, node, i):
		if i == n: # base case
			return [prefix] if 0 in node else []

		r = [] # empty list gets returned if there's no way to keep traversing
		for l in d[digits[i]]:
			if l in node:
				r += traverse(prefix+l, node[l], i+1)
		return r

	return traverse("", dictionary, 0)

# I'm assuming I get my list of words as a prefix tree, which is usually drawn as nodes with
# values = the path used to get to each one, but I'm assuming each node kind of just wraps a map,
# so I can give it as a bunch of nested maps. I'm letting my end node be 0.
dictionary = {'t':{'r':{'e':{'e':{0:0}}}}, 'u':{'s':{'e':{0:0,'d':{0:0}}}}}
assert set(twenty(8733, dictionary)) == set(['tree', 'used'])
assert twenty(873, dictionary) == ['use']
assert twenty(92783, dictionary) == []

def twentyone(A, B):
	# gotta do some algebra: We know (a1 + a2 + a3 + ... + ax ...) - (b1 + b2 + b3 + ... + bx ...) = diff
	# We can choose an ax and bx to swap such that the sums are equal, so:
	# (a1 + a2 + a3 + ... + bx) - (b1 + b2 + b3 + ... + ax) = 0. Subtract the two equations:
	# ax - bx - (bx - ax) = diff - 0 -> 2ax - 2bx = diff -> ax - bx = diff/2. With this we can
	# find the matching ax for bx or vice versa if we have a candidate in mind. Checking existence
	# is cheap.
	c = Counter(A)
	diff = sum(A) - sum(B) # O(|A| + |B|)
	if diff % 2 != 0: return "impossible"
	
	for bx in B: # O(|B|)
		ax = diff//2 + bx
		if ax in c:
			return ax, bx

	return "impossible"

assert twentyone([4,1,2,1,1,2], [3,6,3,3]) == (1, 3)
assert twentyone([4,1,2,1,1,2], [3,6,3,4]) == "impossible"
assert twentyone([100, 3], [1, 100]) == "impossible"

