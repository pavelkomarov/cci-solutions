
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

def seven(a, b):
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




