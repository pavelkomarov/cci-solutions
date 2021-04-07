print("==one==")
# For each of our 20 bottles, put different numbers of pills in each one. The first gets 1,
# next 2, and so on. The total weight should come out to be 1+2+3+...20 if we have only
# 1.0g pills = 20*(21)//2 grams
print(20*21//2) # = 210
# If the first bottle has the heavy pills, then the result will be 210.1, if it's the second,
# then 210.2 ... 212.0 for the last bottle.

print("==two==")
# game 1: probability of winning = p
# game 2: probability of winning = (3 choose 2) * p^2 * (1-p)^1 + (3 choose 3) * p^3 * (1-p)^0
import math

for p in range(10):
	p /= 10
	p2 = math.comb(3,2)* p**2 * (1-p) + p**3
	print(p, p2)

# It's at 0.5 that p2 overtakes p1. I at first thought p2 would always be greater than p1, but I
# guess it makes sense, because if your likelihood of making one shot is low, then p^2 and p^3 will
# be tiny. We can also solve via algebra: p = 3p^2(1-p) + p^3 -> 0 = 2p^2 - 3p + 1 -> solutions at
# 1/2 and 1. It's easy enough to then check around these points to discover p1, p2 -> 1 together, and
# p2 > p1 on the interval between 1/2 and 1, and p1 > p2 on the interval below 1/2. Because they
# can't cross in these intervals, a single check in each is sufficient.

#print("==three==")
# not possible, because a domino must cover one white square and one black square, but if we chew
# off opposite corners, they have the same color. So however we place dominoes, we end up with
# two white or two black left, and you can't cover them with the last domino, because they can't
# be adjacent.

#print("==four==")
# There's going to be a collision unless they all pick the same direction to walk. That's 000 or 111
# on a triangle, and there are 2^3 possibilities there, so that's 2/(2^3) = 1/(2^2). For n-polygons
# it should scale to 1/(2^(n-1)). So Pr[collision] = 1 - 1/(2^(n-1))

#print("==five==")
# I can get 8 by filling and pouring in the 5 and the 3. Similarly I can get 2 by pouring in the 5
# and taking out 3. Repeat to get 4.
# Above I was assuming there's a reservoir we're pouring in to that we can take from. There's also a
# way without such a reservoir: Fill 5, fill 3 from 5. Dump 3. Fill 3 from what remains in 5 (2).
# Fill 5, fill what remains to be filled in 3 (1) from 5. There are now 4 left in 5.

#print("==six==")
# If only one person has blue eyes, then he sees no one else does and leaves on day 1. If two people
# do, then they see each other, assume that's the person with the blue eyes, and stay. But then
# seeing the other still there the next day, they conclude that they came to that same conclusion
# and therefore there is at least one other person with blue eyes, so it must be themselves, so both
# leave day 2. If three, then they see each other, assume that's all, then the other two don't leave
# the next day, so it's day 3 the third person realizes the third has got to be them and they leave.
# If four, then each assumes there are three, but doesn't see anyone leave that day, concludes four,
# leaves day 4. So num days = num people with blue eyes.

print("==seven==")
# 50% of the time you get a girl and stop, say string 0
# 50% of what remains you get a boy, then a girl and stop, say 10
# 50% of remaining you get 110
# ... on a and on
# So that's 1 girl in 100% of cases, 0 boy in 50% of cases, 1 boy in 25% of cases, 2 boys in 12.5%
# of cases ...
# 0*1/2 + 1*1/4 + 2*1/8 + 3*1/16 = sum k = 0 to infinity k/(2^(k+1)) = 1, so the ratio of the new
# generation will still be evenly split boys and girls
import random

b = 0
g = 0
for i in range(10000):
	r = random.randint(0,1)
	while r == 1:
		b += 1
		r = random.randint(0,1)
	g += 1

print(b, g)
# This makes intuitive sense after I see it coded. Basically you have a big random string of 0s
# and 1s, where the probability of each is independent and 50/50. You basically split the string
# up at each 0 by imposing the apocalypse queen rule, but you don't change its contents.

print("==eight==")
# classic egg drop. The unfortunate thing about this problem is it's always phrased as "You have
# N floors and d drops" or "You have N floors and k eggs" "How many eggs/drops does it take in
# the worst case to find f, the first floor where the egg breaks/last one where it doesn't?",
# which is just confusing. Really, it's better to think of N as the output and k,d as the input:
# "How many floors can I guarantee I cover with k eggs and d drops?"
#
# If I have 0 eggs or 0 drops, then I can cover no floors, so that's our base case. If I drop and
# it breaks, I still need to find the last floor where an egg doesn't to be sure of where the
# cutoff lies. If the egg doesn't break, then I need to find the first one where it does to be sure
# of the cutoff. If the egg breaks, I've effectively covered everything above, because I know what
# will happen if I drop from higher. And if it doesn't, I cover everything below, because I know
# what will happen if I drop from lower. So I've either got k-1 or k eggs, and I have d-1 drops,
# and since the ranges I'll be searching in those two cases don't overlap, the number of floors
# I can cover at the dth drop is perfectly additive. One more wrinkle: By dropping at a floor, I
# cover that floor, so we get a +1 in the recurrence.
table = [0,0,0,0] # k=0,1,2 here, d = 0
d = 0
while table[-1] < 100:
	next_row = [0, 1+table[0]+table[1], 1+table[1]+table[2]]
	table = next_row
	d += 1
print(d)

# Alternative math-based solution for exactly two eggs: We want to balance their drops so
# |egg1 drops| + |egg2 drops| = d. As egg1 survives more drops, egg2 should have 1 fewer floors to
# cover. Say egg1 breaks on the first drop from floor x, then egg2 will have to be dropped from 1
# up to x-1 in the worst case. 1 + x-1 = d -> x = d. So here |egg1 drops| = 1, and |egg2 drops| = d-1
# Say egg1 breaks on the second drop, then for our balance to hold, we'll want egg2 to only need to
# cover d-2 locations: d+1 (right above last egg1 drop) up to d+(d-2). This means we should drop egg1
# at d+(d-1). This ladder continues upward until our egg1 reaches 100:
# d + (d-1) + (d-2) + (d-3) ... 2 + 1 = 100. We can sum up the left side with Euler's trick:
# d*(d+1)//2 = 100, and if we take the ceiling of the solution, we get d = 14.
d = (-0.5 + math.sqrt(0.5**2 - 4*0.5*(-100)))/(2*0.5)
print(math.ceil(d))

print("==nine==")
l = [0]*100
for i in range(0,100):
	for j in range(i,100,i+1):
		l[j] ^= 1

# 1 gets toggled at flip 1. 2 gets toggled at flips 1,2. 3 gets toggled at 1,3. In general a location
# is toggled at each of its factors. So the problem really reduces to "How many numbers in [1,100] have
# an odd number of factors?" Insight: The number of factors a number has is always even, unless one of
# the factors is repeated, making the number a perfect square!
import numpy
print(numpy.where(numpy.array(l)==1)[0] + 1)

# There are, of course, 10 perfect squares <=100, because 100 is 10^2
i = 1
while i*i <= 100:
	i += 1
print(i-1)

print("==ten==")
# This is very similar to the error correction puzzles/codes 3blue1brown has videos about.
# https://www.youtube.com/watch?v=b3NxrZOu_CE
# 1000 bottles is just under 32^2 = 1024, so array the bottles in a 32x32 grid. Put drops from the
# 512(ish) bottles from the right side on test strip 1. Then test strips 2, 3..5 get ever finer
# vertical stripes. Then strip 6 gets the lower 512(ish) bottles, and strips 7..10 get ever finer
# horizontal stripes.
# If we had only 4 bits, it would look like
# +------+------+------+------+
# | 0000 | 0001 | 0010 | 0011 |
# +------+------+------+------+
# | 0100 | 0101 | 0110 | 0111 |
# +------+------+------+------+
# | 1000 | 1001 | 1010 | 1011 |
# +------+------+------+------+
# | 1100 | 1101 | 1110 | 1111 |
# +------+------+------+------+
# Let's call bits [4,3,2,1]. Strip1 is everywhere bit2 is 1. Strip2 is everywhere bit1 is 1. Strip3
# is everywhere bit 4 is 1, and strip4 is everywhere bit 3 is 1. You can see these cover all the bits.
#
# We're going to get back some positive and some negative tests, and that will allow us to laser in
# on which bits are set and which aren't in our poisoned bottle's location.
poisoned = random.randint(1,1000)

table = numpy.zeros((32,32))
r,c = divmod(poisoned-1,32) # zero-index for location in table
table[r,c] = 1

# sum of 1 means the poison was included, sum of 0 means it wasn't
answer = 0
for i in range(10):
	strip = sum([table[divmod(x,32)] for x in range(1024) if (x >> i) & 1])
	if strip:
		answer |= (1 << i)
answer += 1 # to get back to 1-indexed

print(poisoned, answer)
# So it can be done in a single batch of tests and only takes as long as it takes to get them back
