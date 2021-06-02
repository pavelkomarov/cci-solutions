"""Return the numbers from a range in random order. What if we don't actually need very many numbers
relative to the size of the range? How can we save memory?
"""

# Idea: We have the range 10..20. We call randint and get 15. Instead of flipping
# shuf[-1] and shuf[15] and then resampling from shuf, let's say we have seen = {}.
# Let seen[15] = 20, and return 15. Now we resample on 10..19, and say we get 15
# again. So now we return 20 and let seen[15] = 19. Now say we resample 10..18 and
# get 12. We return 12 and let seen[12] = 18. Now we sample 10..17 and get 17,
# we can just return it, because it's the last thing in the range, so no notional
# flipping. Say we do the same again and get 16. Then we sample and get 15; we
# return 19.
#
# Is there some chaining in here that I'm missing? But basically it means we only
# store the "corrupted" parts of the range, and otherwise we just use the range itself.

import random

class Random:
    def __init__(self, lo, hi):
        self.seen = {} # O(what we've seen before) space
        self.lo = lo
        self.i = hi

    def randint(self): # O(1)
        if self.i < self.lo: raise ValueError("you've called randint too many times")
    
        j = random.randint(self.lo, self.i) # i = 20, j = 15, say
        t = self.seen[j] if j in self.seen else j
        
        new_target = self.i
        if self.i in self.seen:
            new_target = self.seen[self.i] # chaining
            del self.seen[self.i]
        self.seen[j] = new_target
        
        if self.i in self.seen: # cleanup to save memory
            del self.seen[self.i]
            
        self.i -= 1
        return t
        
r = Random(10,20)
for i in range(11):
    print(r.randint())

    
