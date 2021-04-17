
# One
# A thread shares the same resources as other threads. Processes don't share resources. A process
# can have many threads, but a thread can belong to only one process. It's convenient to use
# parallel processes if you have a job that can be truly split, which doesn't require sharing of
# any kind, because you don't have to worry about processes interfering. You'd want to make that
# complexity tradeoff and risk threads stepping on each other if you truly do need them to share
# information, because sharing a memory space is a far faster way to share than to write to/read
# from disk or pipe information between processes (by copying it from one's memory space to
# anothers').

# Two
# In order for context to switch, the system has to save the state of the current process and
# restore the saved state of some other process. We really want to know when one process ends and
# another begins. We need processes to log this informtion somehow. But if we have processes write
# this info to disk, then since the timescale of doing so is so large, we could throw the length
# of our measurement significantly, rendering it useless. I'm not sure whether there's so much of
# a delay to print things to a terminal, or to a particular memory location. We'd incur some
# slowdown by doing that extra work, but it's on the scale of work we're doing already, so it
# won't introduce a ton of error. The solutions rightly point out that there will be a lot of
# extraneous stuff going on on a computer, so it's possible that we're interrupted by any number
# of things. I like the suggestion of ping-ponging a piece of data back and forth between processes.
# They'll each have to suspend and wait for the other to be context-switched to, and revived to
# respond. We can also get many time-points from this, which should give us a more reliable
# statistic. The scheduler, which is a bit of OS black magic beyond our direct control, will also
# be encouraged to prioritize the correct next process. The solutions point out another wrinkle:
# sending and recieving will take time, but we can figure out how much by sending and recieving a
# bit of data to the same process over the pipe or network and then subtract it out.

# Python's threads don't really work in parallel, but for purposes of these questions, I think
# they're sufficient to implement the spirit of the answer.
# https://docs.python.org/3/library/threading.html
# https://realpython.com/intro-to-python-threading/
from threading import Thread, Lock
from time import sleep

def three(philosophers, race_condition=True):
	N = len(philosophers)
	
	if race_condition: # chaotic, everyone trying to eat at once
		for i in range(N):
			philosophers[i].start()
		for i in range(N):
			philosophers[i].join() # make sure all finish, before checking all are full
	
	else: # A way around deadlock is to have philosophers take turns eating
		for i in range(N): # Even-placed philosophers can eat, except for the last one if he has 0 as a neighbor
			if i % 2 == 0 and i != N-1:
				philosophers[i].start()
				last_started = i
		philosophers[last_started].join() # so this main thread doesn't keep going until they're done eating

		for i in range(N): # Then odd-placed philosophers
			if i % 2 == 1:
				philosophers[i].start()
				last_started = i
		philosophers[last_started].join()

		if not philosophers[-1].full: philosophers[-1].start() # Last one eats if he had 0 as a neighbor

class Philosopher(Thread):
	def __init__(self, i):
		Thread.__init__(self)
		self.i = i

	def run(self):
		# A neat way through the chaos of everyone trying to eat at once is to have a thread relinquish
		# its first lock if it can't get ahold of the second, then try again. Due to jostle in the
		# system, it works. Note that because all of this code gets run in the same memory space as the
		# main code, it has direct access to `chopsticks`, `N`, `verbose`, and `eaten`
		while True:
			if chopsticks[self.i].acquire(False):
				if verbose: print("philosopher",str(self.i),"picked up left chopstick"); sleep(0.2)
				if chopsticks[(self.i+1)%N].acquire(False):
					if verbose: print("philosopher",str(self.i),"picked up right chopstick")
					break
				else: # to experience deadlock, comment out this block (and set verbose=True to get a little waiting time)
					chopsticks[self.i].release()
					if verbose: print("philosopher",str(self.i),"set down left chopstick")

		if verbose: print("philosopher",str(self.i),"is eating")
		eaten[self.i] = True
		if verbose: sleep(0.2); print("philosopher",str(self.i),"is setting down the left chopstick")
		chopsticks[self.i].release()
		if verbose: sleep(0.2); print("philosopher",str(self.i),"is setting down the the right chopstick")
		chopsticks[(self.i+1)%N].release()
		if verbose: print("philosopher",str(self.i),"hath eaten")

N = 5 # N has to be at least 2, or the single philosopher can't eat with the single chopstick
verbose = False
eaten = [False]*N
chopsticks = [Lock() for i in range(N)]
philosophers = [Philosopher(i) for i in range(N)]
three(philosophers, race_condition=True)
assert all(eaten)

from collections import defaultdict

class Four:
	def __init__(self):
		self.locks = []
		self.graph = defaultdict(set)

	def register(self, lock):
		self.locks.append(lock)
		return len(self.locks)-1 # return this new lock's identifier

	def request(self, lock_ndxs):
		if any(not 0 <= i < len(self.locks) for i in lock_ndxs):
			raise ValueError("referencing a lock that does not exist")

		new_edges = set()
		for i in range(len(lock_ndxs)-1):
			# create edge (v,w) if does not already exist
			v = lock_ndxs[i]
			w = lock_ndxs[i+1]
			if w not in self.graph[v]:
				new_edges.add((v, w))
				self.graph[v].add(w)

		# Look for cycle, using graph coloring algorithm
		white = {l for l in self.graph} # nodes get removed as we explore
		for u in lock_ndxs: # for generalized, you'd do this for all nodes in the graph, but here only need the added ones
			if u in white and self._find_cycle(white, set(), u): # no nodes are grey to start, so empty set
				# Then cycle found! Remove new edges, and return False immediately
				for v,w in new_edges:
					if len(self.graph[v]) > 1: self.graph[v].remove(w)
					else: del self.graph[v] # I don't want empty entries, just for cleanliness
				return False
		# If we make it this far, no cycles, and we're good.
		return [self.locks[i] for i in lock_ndxs]

	def _find_cycle(self, white, grey, u):
		# Finding cycles is stupid tricky. Took me quite a while, but finally this article
		# saved me: https://www.geeksforgeeks.org/detect-cycle-direct-graph-using-colors/
		white.remove(u) # change color from white to grey (unexplored -> exploring)
		grey.add(u)

		for v in self.graph.get(u, []): # get() so the defaultdict doesn't create empty entries for sink nodes
			if v in grey:
				return True

			if v in white and self._find_cycle(white, grey, v):
				return True

		grey.remove(u) # change color from grey to black (exploring -> explored)
		return False	# Not keeping black set, because it's not important.

safety_third = Four()
for i in range(10): safety_third.register(Lock())
A = [1,2,3,4]
B = [1,3,5]
C = [7,5,9,2]
assert safety_third.request(A)
assert str(safety_third.graph) == "defaultdict(<class 'set'>, {1: {2}, 2: {3}, 3: {4}})"
assert safety_third.request(B)
assert str(safety_third.graph) == "defaultdict(<class 'set'>, {1: {2, 3}, 2: {3}, 3: {4, 5}})"
assert not safety_third.request(C)
assert str(safety_third.graph) == "defaultdict(<class 'set'>, {1: {2, 3}, 2: {3}, 3: {4, 5}})"
try:
	safety_third.request([10,12])
	assert False
except ValueError:
	assert True



