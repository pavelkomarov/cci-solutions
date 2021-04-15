
def one(N, M, i, j):
	mask = ((1 << j-i+1) - 1) << i # gives me j-i+1 1s to fill [i,j]
	return (N & ~mask) | (M << i)

N = 1076
M = 19
i = 2
j = 6
assert one(N, M, i, j) == 1100

def two(d):
	b = 1/2
	s = "0."
	while d > 0 and len(s) < 34:
		if d >= b:
			s += "1"
			d -= b
		else:
			s += "0"
		b /= 2

	if d > 0: raise ValueError("can't represent that number in 32 significant digits")
	return s

assert two(0.5) == "0.1"
assert two(0.25) == "0.01"
assert two(0.125) == "0.001"
assert two(0.75) == "0.11"
try:
	two(0.8)
	assert False
except ValueError:
	assert True

def three(n):
	prev = 0 # length of previous group
	cur = 0 # length of current group
	m = 0
	for i in range(32):
		b = (n >> i) & 1
		if b: # if the ith bit is a 1
			cur += 1
		else:
			m = max(m, prev+cur+1) # first round prev = 0
			prev = cur
			cur = 0

	return max(m, prev+cur+1) # in case last bit isn't a 0, this won't have been triggered

y = -1 ^ (1 << 4)
assert three(y) == 32
assert three(1775) == 8
y = ((1 << 31) - 1) ^ (1 << 4)
assert three(y) == 31
y = ((1 << 31) - 1) ^ (3 << 4)
assert three(y) == 31 - 5 # because one flip can't span the two 0s.

def four(n):
	# gen next-largest: first 0 to left of 1s gets filled, and all
	# 1s right of that get shifted to least-significant positions
	ones = 0
	left_zero = False
	i = -1
	while not left_zero:
		i += 1
		b = (n >> i) & 1
		if b: ones += 1
		if not b and ones > 0: left_zero = True
	# i is now wherever the left_zero is
	l = n
	for j in range(i):
		if ones > 1:
			l |= (1 << j)
			ones -= 1
		else: l &= ~(1 << j)
	l |= (1 << i)

	# gen next-smallest: first 0 right of a 1 gets filled, and all
	# 1s right of that get shifted to most-significant positions
	ones = 0
	right_zero = False
	i = -1
	while True:
		i += 1
		b = (n >> i) & 1
		if b:
			ones += 1
			if right_zero: break
		else: right_zero = True

	s = n
	for j in range(i-1):
		if j < i-ones: # unset first bits
			s &= ~(1 << j)
		else: s |= (1 << j) # set last bits
	s &= ~(1 << i)
	s |= (1 << i-1)

	return l,s

# Do this the dumb way just for testing
def four_brute_force(n):
	ones = bin(n).count('1')
	l = n+1
	while bin(l).count('1') != ones:
		l += 1
	s = n-1
	while bin(s).count('1') != ones:
		s -= 1
	return l,s

for i in range(2, 100):
	if i in [3, 7, 15, 31, 63]: continue # infinite loops if no rightward zeros
	assert four_brute_force(i) == four(i)

def five(n):
	return n & (n-1) == 0

# It checks whether n is a power of 2 or 0
for i in range(-100, 100):
	if i in [0,1,2,4,8,16,32,64]: # include 0 also
		assert five(i)
	else:
		assert not five(i)

def six(A, B):
	X = A ^ B # just count up the number of places they differ
	c = 0
	for i in range(32):
		if (X >> i) & 1:
			c += 1
	return c

assert six(29, 15) == 2

def seven(n):
	# Rather than search for these decimal digits, you can write hex:
	# 0xaaaaaaaa for the odd bits, 0x55555555 for the even bits
	odd = n & -1431655766 # odd bits set
	even = n & 1431655765 # even bits set
	return (even << 1) | (odd >> 1)

assert seven(42) == 21

def eight(screen, w, x1, x2, y):
	start = y*w + x1 # let's assume x and y are zero-indexed
	end = y*w + x2 # I'm setting [start, end)

	sbyte, sbits = divmod(start, 8)
	ebyte, ebits = divmod(end, 8)

	# First byte gets the remainder of the start divmod set bits on its least-
	# significant side
	left = (1 << sbits) - 1
	# Last byte gets 8-remainder of the end divmod bits set on its most-significant side
	right = (~((1 << 8-ebits) - 1)) & ((1<<8)-1)

	if sbyte == ebyte:
		screen[sbyte] |= left & right
	else:
		screen[sbyte] |= left
		# If we're drawing a long line, then fully set bytes in between
		for i in range(sbyte+1, ebyte):
			screen[i] = 255
		screen[ebyte] |= right

screen = bytearray(b'\x00'*8*64) # 4096 pixels all together
w = 64
eight(screen, w, 4, 5, 3)
assert screen[20:30] == bytearray(b'\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00')
eight(screen, w, 4, 30, 3)
assert screen[20:30] == bytearray(b'\x00\x00\x00\x00\x0f\xff\xff\xfc\x00\x00')
eight(screen, w, 4, 32, 3)
assert screen[20:30] == bytearray(b'\x00\x00\x00\x00\x0f\xff\xff\xff\x00\x00')
