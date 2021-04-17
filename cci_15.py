
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
# won't introduce a ton of error.

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
	def __init__(self, i, left_chopstick, right_chopstick, verbose=False):
		Thread.__init__(self)
		self.full = False
		self.i = i
		self.left_chopstick = left_chopstick
		self.right_chopstick = right_chopstick
		self.verbose = verbose

	def run(self):
		# A neat way through the chaos of everyone trying to eat at once is to have a thread relinquish
		# its first lock if it can't get ahold of the second, then try again. Due to jostle in the
		# system, it works.
		while True:
			if self.left_chopstick.acquire(False):
				if self.verbose: print("philosopher",str(self.i),"picked up left chopstick"); sleep(0.2)
				if self.right_chopstick.acquire(False):
					if self.verbose: print("philosopher",str(self.i),"picked up right chopstick")
					break
				else:
					self.left_chopstick.release()
					if self.verbose: print("philosopher",str(self.i),"set down left chopstick")

		if self.verbose: print("philosopher",str(self.i),"is eating")
		self.full = True
		if self.verbose: sleep(0.2); print("philosopher",str(self.i),"is setting down the left chopstick")
		self.left_chopstick.release()
		if self.verbose: sleep(0.2); print("philosopher",str(self.i),"is setting down the the right chopstick")
		self.right_chopstick.release()
		if self.verbose: print("philosopher",str(self.i),"hath eaten")

N = 5 # N has to be at least 2, or the single philosopher can't eat with the single chopstick
chopsticks = [Lock() for i in range(N)]
philosophers = [Philosopher(i, chopsticks[i], chopsticks[(i+1)%N], verbose=False) for i in range(N)]
three(philosophers, race_condition=True)
assert all(p.full for p in philosophers)

def four():
	pass



