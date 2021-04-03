
class One:
	# I'm just not allowing stacks over size//3. I don't think this problem
	# really merits moving things around and such. It's not an interesting
	# enough data structure.

	def __init__(self, size):
		self.size = size
		self.arr = [0]*size
		self.stack1 = 0
		self.stack2 = size//3
		self.stack3 = 2*size//3

	def push(self, which, el):
		# just doesn't do anything if you exceed size limit
		if which == 1:
			if self.stack1 < self.size//3:
				self.arr[self.stack1] = el
				self.stack1 += 1

		elif which == 2:
			if self.stack2 < 2*self.size//3:
				self.arr[self.stack2] = el
				self.stack2 += 1

		elif which == 3:
			if self.stack3 < self.size:
				self.arr[self.stack3] = el
				self.stack3 += 1

	def pop(self, which):
		if which == 1:
			if self.stack1 > 0:
				self.stack1 -= 1
				return self.arr[self.stack1]

		elif which == 2:
			if self.stack2 > self.size//3:
				self.stack2 -= 1
				return self.arr[self.stack2]

		elif which == 3:
			if self.stack3 > 2*self.size//3:
				self.stack3 -= 1
				return self.arr[self.stack3]

stacks = One(10)
for i in range(5,9):
	stacks.push(1,i)
for i in range(7,4,-1):
	assert stacks.pop(1) == i
assert stacks.pop(1) is None

for i in range(5,9):
	stacks.push(3,i)
for i in range(8,4,-1):
	assert stacks.pop(3) == i
assert stacks.pop(3) is None


class Two:
	def __init__(self):
		self.arr = []
		self.mins = [] # wherever minimums change as we go up the stack

	def push(self, el):
		if el < self.min():
			self.mins.append(el)
		self.arr.append(el)
		
	def pop(self):
		el = self.arr.pop()
		if self.min() == el:
			self.mins.pop()
		return el

	def min(self):
		return self.mins[-1] if len(self.mins) > 0 else float('inf')

stack = Two()
stack.push(10)
stack.push(7)
stack.push(8)
stack.push(1000)
assert stack.min() == 7
assert stack.pop() == 1000
assert stack.pop() == 8
assert stack.pop() == 7
assert stack.min() == 10


class Three:
	def __init__(self, threshold):
		self.threshold = threshold
		self.stacks = [[]]

	def push(self, el):
		if len(self.stacks[-1]) == self.threshold:
			self.stacks.append([el])
		else:
			self.stacks[-1].append(el)

	def pop(self):
		if len(self.stacks[-1]) == 1:
			return self.stacks.pop()[0]
		else: return self.stacks[-1].pop()

	def popAt(self, i):
		if len(self.stacks[i]) > 1:
			return self.stacks[i].pop()
		else:
			return self.stacks.pop(i)[0]

stack = Three(2)
for i in range(10):
	stack.push(i)
assert stack.pop() == 9
assert stack.pop() == 8
assert stack.popAt(1) == 3
assert stack.popAt(1) == 2
assert stack.popAt(1) == 5


class Four:
	def __init__(self):
		self.stack1 = []
		self.stack2 = []

	def add(self, el):
		while len(self.stack2) > 0:
			self.stack1.append(self.stack2.pop())
		self.stack1.append(el)

	def remove(self):
		while len(self.stack1) > 0:
			self.stack2.append(self.stack1.pop())
		return self.stack2.pop()

q = Four()
for i in range(5):
	q.add(i)
assert q.remove() == 0
assert q.remove() == 1
q.add(5)
assert q.remove() == 2


def five(stack):
	# Assume pile is in order. We pop an element off the stack, and we want to insert
	# it at the proper location in pile. We pop off all the elements of pile that are
	# > the element, place the element, and then move stuff back over top of it. We
	# repeat until the stack is empty and we've built pile all the way up. Then we just
	# reload everything from the pile in to the stack.
	pile = []
	
	while len(stack) > 0:
		el = stack.pop()
		while len(pile) > 0 and el < pile[-1]:
			stack.append(pile.pop())
		pile.append(el)

	while len(pile) > 0:
		stack.append(pile.pop())

stack = [1,2,5,3,4]
five(stack)
assert stack == [5,4,3,2,1]


from cci_2 import Node

class Six:
	def __init__(self):
		self.cats = None
		self.dogs = None
		self.i = 0

	def enqueue(self, name, species):
		if species == 'cat':
			if self.cats is None:
				self.cats = Node((name, self.i))
			else:
				node = self.cats
				while node.next is not None:
					node = node.next
				node.next = Node((name, self.i))

		elif species == 'dog':
			if self.dogs is None:
				self.dogs = Node((name, self.i))
			else:
				node = self.dogs
				while node.next is not None:
					node = node.next
				node.next = Node((name, self.i))

		self.i += 1

	def dequeueAny(self):
		oldest_cat = self.cats.val[1] if self.cats is not None else float('inf')
		oldest_dog = self.dogs.val[1] if self.dogs is not None else float('inf')
		if oldest_cat < oldest_dog:
			return self.dequeueCat()
		else:
			return self.dequeueDog()

	def dequeueCat(self):
		cat = self.cats.val[0]
		self.cats = self.cats.next
		return cat

	def dequeueDog(self):
		dog = self.dogs.val[0]
		self.dogs = self.dogs.next
		return dog

shelter = Six()
shelter.enqueue('Pumpkin', 'cat')
shelter.enqueue('Fozzy', 'cat')
shelter.enqueue('Bo', 'dog')
shelter.enqueue('Benny', 'dog')
assert shelter.dequeueAny() == 'Pumpkin'
assert shelter.dequeueDog() == 'Bo'
assert shelter.dequeueCat() == 'Fozzy'
assert shelter.dequeueAny() == 'Benny'
