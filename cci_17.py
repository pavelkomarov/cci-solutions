
def one(A, B):
	# A and B are composed of bits, so we can do a ripple-add if we keep an extra carry bit and
	# draw a K-map to figure out what the logic should be. Time for some ASCII art!
	# Call the addition result bit x, call the carry bit c, and call bits from A and B, a and b.
	#
	#      \ ab                                    \ ab
	#     c \ _00__01__11__10                     c \ _00__01__11__10
	#      0 | 0 | 1 | 0 | 1 |                    0 | 0 | 0 | 1 | 0 |
	#        |---------------| = x                  |---------------| = c'
	#      1 | 1 | 0 | 1 | 0 |                    1 | 0 | 1 | 1 | 1 |
	#        '---------------'                      '---------------'
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

