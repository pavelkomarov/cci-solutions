"""The prompt was to find the largest prime made of all unique digits from [1,9]
"""

# first some inspiration: some sieve of Erotosthenes code
"""
if n < 2: print(0)

l = [i for i in range(2,n)]
s = set(l)
primes = set()

for m in l:
		if m in s:
				primes.add(m)
				c = m
				while c < n: # remove all multiples of m
						s.discard(c)
						c += m

print(len(primes))
"""
# Wow, I feel inspired.


from itertools import permutations
import numpy

# first get a list of all "monomials" in order
digits = [[str(i) for i in range(end,0,-1)] for end in range(9,1,-1)]
print(digits)

mons_dec = []
for dset in digits:
	perms = permutations(dset)
	mons_dec += [int(''.join(p)) for p in perms]

assert numpy.all(numpy.diff(mons_dec) < 0)
print("len(mons_dec)", len(mons_dec))


# The idea now is to search this list for the largest member
# that makes isPrime true. First I'll need an isPrime.
# I've shamelessly ripped this from stack overflow.
def is_prime(n):
	if n == 2 or n == 3: return True
	if n < 2 or n%2 == 0: return False
	if n < 9: return True
	if n%3 == 0: return False
	r = int(n**0.5)
	# since all primes > 3 are of the form 6n ± 1
	# start with f=5 (which is prime)
	# and test f, f+2 for being prime
	# then loop by 6. 
	f = 5
	while f <= r:
		#print('\t',f)
		if n % f == 0: return False
		if n % (f+2) == 0: return False
		f += 6
	return True


assert is_prime(7)
assert not is_prime(100)
assert not is_prime(1000000)
assert is_prime(17389)
assert is_prime(2750159)


for mono in mons_dec:
	if is_prime(mono):
		print("largest prime monomial:", mono)
		break


# I thought at one point I'd find use for a binary search, but primes don't follow a pattern
# where you can sort the unprime from the prime cleanly. Just more inspiration.
"""
l = 0
r = len(mons_inc)-1

while l < r:
    mid = (l + r) // 2
    
    #is_prime()
   
    if s < m:
        l = mid + 1
    else:
        r = mid
        
return (l + r)//2
"""
