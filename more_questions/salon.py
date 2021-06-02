"""The prompt was to simulate a Salon, with a bunch of very particular rules about what happens under
different conditions
"""

import numpy
import sys
from io import StringIO
from collections import defaultdict

# I'm going to simulate 10 hours, from 8am to 6pm. If customers arrive before opening, they also curse themselves.
# I'm modeling the arrival of customers with a Poisson process with lambda 1/10. I call to decide how many new
# customers to generate each minute This gives me an expected value of 1 customer every 10 minutes but allows
# customers to arrive randomly, even at the same time as others.

class Customer:
	"""Helper class to keep a universal count of customer id-number and print nicely
	"""
	n = 1
	patience = 30 # customers will wait for up to 30 minutes
	
	def __init__(self):
		self.n = Customer.n
		Customer.n += 1

	def __repr__(self):
		return "Customer-" + str(self.n)


class Stylist:
	"""Helper class so I can print these things nicely and keep a few variables.
	"""
	def __init__(self, name: str):
		self.name = name
		self.ocupado_countdown = 0 # How long the stylist has left to work on their current Customer
		self.customer = None # The Customer currently in the Sylist's care
		self.shift_done = False # A flag to make sure Stylists go home

	def __repr__(self):
		return self.name
		
	def assign(self, c: Customer):
		"""A Stylist gets paired with a Customer, and we figure out how long they're going to take.
		"""
		self.customer = c
		self.ocupado_countdown = numpy.random.randint(20, 40)
	

class Time:
	"""I'm making this iterable class with a bunch of fancy magic methods to simplify looping through the day and
	checking shift ends.
	"""
	def __init__(self, start: str='9am', end: str='6pm'):
		self.t = self._parse_time(start) # the number of minutes past 9:00am
		self.f = self._parse_time(end) # f for "finish" or "final"

	def _parse_time(self, s: str):
		"""I'm considering '9am' to be t = 0. If I give '1pm' I want to return 60*4 = the number of minutes 1pm
		happens after 9am.
		"""
		hour = int(s[:-2])
		if s[-2:] == 'pm':
			hour += 3
		elif s[-2:] == 'am':
			hour -= 9
		else:
			raise ValueError("Can not parse {0}. Must end in 'am' or 'pm'".format(s))
		return hour*60
	
	def __next__(self):
		"""I want Time to be iterable
		"""
		if self.t < self.f:
			self.t += 1
			return self.t
		else:
			raise StopIteration
		
	def __iter__(self):
		self.t -= 1 # because I want to open at 9:00, not 9:01 after I increment t in __next__
		return self
		
	def __ge__(self, other): # defining this makes >= and <= work
		return self.t >= other.t
		
	def __gt__(self, other): # makes > and < work
		return self.t > other.t
		
	def __eq__(self, other): # so == works
		return self.t == other.t
	
	def __repr__(self):
		"""return HH:MM string representation
		"""
		hour, minute = divmod(self.t, 60)
		hour = str(hour + 9)
		minute = str(minute)
		if len(hour) < 2: hour = '0' + hour
		if len(minute) < 2: minute = '0' + minute
		return hour + ':' + minute
		

class Salon:
	opening_time = Time('9am')
	shift_change = Time('1pm')
	closing_time = Time('5pm')
	
	first_shift = [Stylist(name) for name in ['Anne', 'Ben', 'Carol', 'Derek']]
	second_shift = [Stylist(name) for name in ['Erin', 'Frank', 'Gloria', 'Heber']]
	
	def __init__(self, start: str, end: str, lam: float=0.1, capacity: int=15):
		self.lam = lam # lambda parameter for the poisson distribution
		self.capacity = capacity # max customers allowed in the building at once
		self.clock = Time(start, end) # the master clock of the simulation

		self.on_duty = set() # staff currently in house
		self.waiting = [] # holds tuples (Customer, minute they started waiting)
		self.n_customers = 0 # the number of customers in house
		self.open = False

	def run(self):
		"""Save sanity by just considering what little operation have to happen every minute, and call those in
		succession.
		"""
		for minute in self.clock:
			if self.clock == Salon.opening_time:
				self._open()

			# Customers can arrive whether we're open or not
			self._customers_arrive(minute) # in case any *do* arrive, `minute` gives us a notion of when for the waitlist

			# Nothing happens in these three if the salon isn't open, because `on_duty` and `waiting` are empty
			self._finish_customers() # Stylists finish with customers, who then leave happy
			self._start_customers() # They grab new customers from the waiting list
			self._customers_get_tired(minute) # If their patience has expired, customers leave
			
			if self.clock == Salon.shift_change:
				self._shift_change()
			elif self.clock == Salon.closing_time:
				self._close()

	def _customers_arrive(self, minute: int):
		"""Some new customers randomly show up this minute (according to a Poisson distribution). If outside the Salon's
		hours, they leave cursing themselves. If there are too many customers already in the building, they leave
		impatiently. Otherwise they stand in line.
		"""
		for i in range(numpy.random.poisson(self.lam)): # Poisson gives us the number who should arrive in this minute
			c = Customer()

			if self.clock > Salon.closing_time or self.clock < Salon.opening_time:
				print(self.clock, c, "left cursing themselves") # These guys never "enter", because the place is closed

			elif self.n_customers < self.capacity:
				self.waiting.append((c, minute))
				self.n_customers += 1
				print(self.clock, c, "entered")

			else:
				print(self.clock, c, "left impatiently")

	def _finish_customers(self):
		"""If a stylist is still working on a customer, just let them keep working. If they're done, then free up that
		stylist and let the customer leave satisfied. If in addition the stylist's shift is over, let them go home.
		"""
		go_home = set()

		for stylist in self.on_duty:
			if stylist.ocupado_countdown > 0: # keep working on customer
				stylist.ocupado_countdown -= 1

			elif stylist.customer is not None: # This one's done!
				print(self.clock, stylist, "ended cutting", str(stylist.customer) + "'s hair")
				print(self.clock, stylist.customer, "left satisfied")

				self.n_customers -= 1 # someone leaves the building
				stylist.customer = None
				
				if stylist.shift_done: go_home.add(stylist)

		if len(go_home) > 0:
			self.__send_staff_home(go_home)
			self.__lock_up_if_last_out()

	def _start_customers(self):
		"""Take customers from the queue and start their haircuts by assigning them to stylists.
		"""
		for stylist in self.on_duty:
			if stylist.customer is None and len(self.waiting) > 0:
				stylist.assign(self.waiting.pop(0)[0])
				print(self.clock, stylist, "started cutting", str(stylist.customer) + "'s hair")

	def _customers_get_tired(self, minute: int):
		"""Customers that have been waiting too long leave unfulfilled.
		"""
		while len(self.waiting) > 0 and minute - self.waiting[0][1] > Customer.patience:
			c = self.waiting.pop(0)[0]
			self.n_customers -= 1
			print(self.clock, c, "left unfulfilled")

	def _open(self):
		"""When the salon opens, we print that we've opened, the first shift comes on duty, and we print that they have
		"""
		self.open = True
		print(self.clock, "Hair Salon opened")

		for stylist in Salon.first_shift:
			self.on_duty.add(stylist)
			print(self.clock, stylist, "started shift")

	def _shift_change(self):
		"""The second shift arrives, so print they have. Mark all the stylists from first shift as free to go home as
		soon as they're done cutting someone's hair, or, if they're already not, let them go home immediately.
		"""
		self.__send_staff_home()

		for stylist in Salon.second_shift: # second shift arrives
			self.on_duty.add(stylist)
			print(self.clock, stylist, "started shift")

	def _close(self):
		"""Do end of day things. Set all the staff in motion to go home, kick customers still waiting out, lock up.
		"""
		self.__send_staff_home()

		for c,t in self.waiting:
			print(self.clock, c, "left furious")
		self.waiting = []
		self.n_customers = 0

		self.__lock_up_if_last_out()

	def __send_staff_home(self, go_home=None):
		"""Send particular staff home, or send a shift home. If a Stylist isn't working on a customer at the end of their shift,
		send them home immediately. Otherwise, mark that they can go home.
		"""
		if go_home is None:
			go_home = set() # modifying the set as it's being iterated can cause undefined behavior, so keep a second set

			for stylist in self.on_duty:
				if stylist.customer is None: # they can go home right away
					go_home.add(stylist)
				else: # they'll go home when they're finished with who they're with
					stylist.shift_done = True

		for stylist in go_home:
			self.on_duty.remove(stylist)
			print(self.clock, stylist, "ended shift")

	def __lock_up_if_last_out(self):
		"""Look around to see if there's anyone still in. If not, close up.
		"""
		if self.open and len(self.on_duty) == 0:
			self.open = False
			print(self.clock, "Hair Salon closed")


stream = StringIO()
sys.stdout = stream # So I can capture the stdout for testing purposes

salon = Salon('9am', '6pm', lam=0.1)
salon.run()

sys.stdout = sys.__stdout__ # get back everything the Salon sim printed
stream.seek(0)
sim_result = stream.read()
print(sim_result)

lines = [l.split() for l in sim_result[:-1].split('\n')] # chop off the last newline character
#print(lines)

# In production I'd use pytest or something, but here I just write bare tests. These are kind of annoyingly specific,
# because they depend heavily on the string format coming out of the simulator. I'd want the simulation result to
# be something a little more structured than printed strings were we going to really rely on its output downstream.
# See my "ensure" comments above each block to understand what it's testing. At the bottom I have some additional
# ideas of what to test, but I've decided I've had enough of this and leave them unimplemented.

# ensure time stamps only go forward
for i in range(len(lines)-1):
	assert lines[i][0] <= lines[i+1][0] # exploit the fact I'm using 24 hour time stamps to just use string comparison

# ensure stylists always have 0 or 1 customers at all times, and
# ensure the same customer gets started and ended on by the same stylist (no weird teleporting salon chairs), and
# ensure 20-40 minutes for all, and
# ensure stylists properly go home -> can't start clients after shift end
c = {stylist.name:[] for stylist in Salon.first_shift+Salon.second_shift}
for line in lines:
	name = line[1]
	time = line[0].split(':')
	time = int(time[0])*60 + int(time[1]) # get time as minutes from midnight

	if line[2:4] == ['started', 'cutting']:
		if time > 60*13: # if it's after 1:00pm, y'all better not be from first shift
			assert name not in Salon.first_shift
		assert time <= 60*17 # starting has to occur at or before 5:00pm

		assert c[name] == []
		c[name].append((time, line[4])) # the stylist with that name now has the customer with name in 4th position
	elif line[2:4] == ['ended', 'cutting']:
		assert len(c[name]) == 1
		assert c[name][0][1] == line[4]
		assert 20 <= time - c[name][0][0] <= 40 # The time started has to be in [20,40] before present
		c[name] = []

# ensure customers are serviced in order
prev_cust_number = 0
for line in lines:
	if line[2:4] == ['started', 'cutting']:
		cust_name = line[4]
		cust_number = int(cust_name[9:-2]) # slice out the digits
		assert cust_number > prev_cust_number
		prev_cust_number = cust_number

# ensure no more than capacity in house at any given time, and that extras leave impatiently
# Shinking n_customers leads to more impatient leaving, but I have witnessed it with limit 15.
c = 0
for line in lines:
	if line[-1] == 'entered':
		c += 1
	elif line[2] == 'left':
		if c == salon.capacity:
			assert line[-1] == 'impatiently'
		elif line[3] != 'cursing':
			c -= 1

	assert 0 <= c <= salon.capacity

# ensure everyone checks in before they check out, and
# ensure that opening happens before first shifts begin, and
# ensure last out locks up
on_duty = defaultdict(int)
openclose = 0
for line in lines:
	if line[-2:] == ['Salon', 'opened']:
		assert sum(on_duty.values()) == 0
		openclose += 3
	elif line[-2:] == ['Salon', 'closed']:
		assert sum(on_duty.values()) == 0
		openclose += 5

	elif line[-2:] == ['started', 'shift']:
		on_duty[line[1]] += 1
		assert on_duty[line[1]] == 1 # No double starting
	elif line[-2:] == ['ended', 'shift']:
		on_duty[line[1]] -= 1
		assert on_duty[line[1]] == 0 # No stopping before starting

assert openclose == 8 # only way this could be 8 is if it hits open condition once and close condition once

# ensure those who arrive outside business hours immediately leave cursing themselves
customers_entered = set()
for line in lines:
	if 'Customer' in line[1]:
		if (line[0] < '09:00' or line[0] > '17:00'):
			if line[1] not in customers_entered:
				assert line[3] == 'cursing'
		else:
			customers_entered.add(line[1])

# More Test ideas!
# ensure customers who wait too long leave unfulfilled. I've seen this, but it's much more common if lambda is increased.
# ensure customers who get a successful haircut leave satisfied
# ensure furious at end if kicked out
