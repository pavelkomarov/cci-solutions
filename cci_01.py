from collections import Counter

def one(s):
	# O(n) time, O(n) space
	c = Counter(s)
	return len(c) == len(s)
	# O(nlogn) time, O(1) space
	#s = sorted(s)
	#for i in range(1, len(s)):
	#	if s[i-1] == s[i]: return False
	#return True

assert one("abc")
assert not one("abb")

def two(a,b):
	return Counter(a) == Counter(b)

assert two("abc", "bca")
assert not two("abc", "dbc")

def three(s):
	s = s.strip()
	new_s = ""
	for c in s:
		if c == ' ':
			new_s += "%20"
		else: new_s += c
	return new_s

assert three("Mr John Smith    ") == "Mr%20John%20Smith"

def four(s):
	# You could use a bit vector to just store parity of each character that occurs
	# rather than use a counter. O(n) space -> O(1) space
	c = Counter(s)
 	x = 0
	for k,v in c.items():
		if v % 2 != 0:
			x += 1
			if x > 1: return False
	return True

assert four("tactcoa")
assert not four("rtactcoa")

def five(a,b):
	n = len(a)
	m = len(b)
	if abs(n-m) > 1: return False
	i = 0
	j = 0
	mismatch = 0
	while i < n and j < m:
		if a[i] != b[j]:
			if n > m: i += 1
			elif m > n: j += 1
			else: i += 1; j += 1
			mismatch += 1
			if mismatch > 1: return False
		else:
			i += 1
			j += 1
	return True

assert five("pale", "ple")
assert five("pales", "pale")
assert five("pale", "bale")
assert not five("pale", "bake")

def six(s):
	# In this implementation I'm leaving off the number if it's just one.
	new_s = s[0]
	c = 1
	for i in range(1, len(s)):
		if s[i] != s[i-1]:
			if c > 1: new_s += str(c)
			new_s += s[i]
			c = 1
		else:
			c += 1
	if c > 1: new_s += str(c)
	return new_s if len(new_s) < len(s) else s

assert six("aabcccccaaa") == "a2bc5a3"

def seven(m):
	# Using O(1) space. The idea is to flip the quarters in to each other
	# I take +90 degrees to mean counter clockwise.
	N = len(m)
	for i in range(N//2):
		for j in range(N//2):
			# there are four places we care about flipping here:
			# (i, j) -> (N-j-1, i) -> (N-i-1, N-j-1) -> (j, N-i-1) -> (i, j)
			v = m[j][N-i-1]
			for a,b in [(i,j), (N-j-1,i), (N-i-1,N-j-1), (j,N-i-1)]:
				t = m[a][b]
				m[a][b] = v
				v = t
	return m

assert seven([[1,2,3,4],
			  [5,6,7,8],
			  [9,10,11,12],
			  [13,14,15,16]]) == [[4,8,12,16],
								  [3,7,11,15],
								  [2,6,10,14],
								  [1,5,9,13]]

def eight(m):
	clear = []
	for i in range(len(m)):
		for j in range(len(m[0])):
			if m[i][j] == 0:
				clear.append((i,j))

	for i,j in clear:
		m[i] = [0]*len(m[0])

	for x in range(len(m)):
		for i,j in clear:
			m[x][j] = 0

	return m

assert eight([[1,2,3,0],
			  [5,6,7,8],
			  [9,0,11,12],
			  [13,14,15,16]]) == [[0,0,0,0],
								  [5,0,7,0],
								  [0,0,0,0],
								  [13,0,15,0]]

def nine(a, b):
	return a in b+b

assert nine("waterbottle", "erbottlewat")
assert not nine("abcde", "cbeab")
