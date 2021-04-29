
class Node: # This ended up getting way more use across way more chapters than I expected.
	def __init__(self, val, next=None, prev=None):
		self.val = val
		self.next = next
		self.prev = prev

	def __repr__(self):
		node = self
		s = ""
		while node is not None:
			arrow = "->" if node.next is None or node.next.prev is None else "<->" # double vs single
			s += str(node.val) + arrow
			node = node.next
		return s

def one(head):
	node = head
	seen = set()
	while node is not None:
		seen.add(node.val)
		if node.next is not None and node.next.val in seen:
			node.next = node.next.next
		node = node.next
	return head
	# If seen isn't allowed, and we want O(1) space, then keep a pointer at a node, and send another
	# forward through the list looking for duplicates of only that value.

ll = Node(10, Node(7, Node(7, Node(2, Node(5, Node(7, Node(3, Node(3))))))))
assert str(ll) == "10->7->7->2->5->7->3->3->"
assert str(one(ll)) == "10->7->2->5->3->"

def two(head, k):
	node = head
	for i in range(k):
		node = node.next
	k_behind = head
	while node is not None:
		node = node.next
		k_behind = k_behind.next
	return k_behind.val

ll = Node(10, Node(7, Node(7, Node(2, Node(5, Node(7, Node(3, Node(3))))))))
assert two(ll, 3) == 7
assert two(ll, 8) == 10

def three(del_node):
	node = del_node
	while node is not None:
		node.val = node.next.val
		if node.next.next is None:
			node.next = None
		node = node.next

assert str(ll) == "10->7->7->2->5->7->3->3->"
c = ll.next.next.next
three(c)
assert str(ll) == "10->7->7->5->7->3->3->"

def four(head, x):
	left = None
	right = None
	nl = None
	nr = None
	node = head

	while node is not None:
		if node.val < x:
			if left is None:
				left = node
				nl = node
			else:
				nl.next = node
				nl = node

		else:
			if right is None:
				right = node
				nr = node
			else:
				nr.next = node
				nr = node

		node = node.next

	# string the halves together
	if left is None: return right
	nl.next = right
	nr.next = None
	return left

ll = Node(3, Node(5, Node(8, Node(5, Node(10, Node(2, Node(1)))))))
assert str(ll) == "3->5->8->5->10->2->1->"
ll = four(ll,5)
assert str(ll) == "3->2->1->5->8->5->10->"

def five(a, b):
	carry = 0
	s = ""
	while a is not None and b is not None:
		carry, digit = divmod(a.val + b.val + carry, 10)
		s += str(digit)
		a = a.next
		b = b.next

	# in case a has extra digits
	while a is not None:
		carry, digit = divmod(a.val + carry, 10)
		s += str(digit)
		a = a.next

	# in case b has extra digits. This won't execute if the above one did.
	while b is not None:
		carry, digit = divmod(b.val + carry, 10)
		s += str(digit)
		b = b.next

	if carry: s += str(carry)
	return int(s[::-1])

a = Node(7, Node(1, Node(6)))
b = Node(5, Node(9, Node(2)))
assert five(a,b) == 912
a = Node(9, Node(9, Node(9, Node(9))))
b = Node(1)
assert five(a,b) == 10000

def six(head):
	tortoise = head
	hare = head
	stack = []
	parity = 0
	while hare is not None:
		stack.append(tortoise.val)

		tortoise = tortoise.next
		hare = hare.next
		if hare is not None: hare = hare.next
		else: parity = 1

	if parity: stack.pop() # odd-length list, so last element in stack is middle element
		# of list, and we don't have to worry about matching against it

	while tortoise is not None:
		if tortoise.val != stack.pop():
			return False
		tortoise = tortoise.next

	return True

ll = Node(1, Node(2, Node(3, Node(4, Node(3, Node(2, Node(1)))))))
assert six(ll)
ll = Node(1, Node(2, Node(3, Node(4, Node(4, Node(3, Node(2, Node(1))))))))
assert six(ll)
ll = Node(1, Node(2, Node(3, Node(4, Node(5, Node(2, Node(1)))))))
assert not six(ll)
ll = Node(1, Node(5, Node(3, Node(4, Node(4, Node(3, Node(2, Node(1))))))))
assert not six(ll)

def seven(a, b, o_1):
	if not o_1:
		# O(n) space solution: Iterate the first, saving its nodes in a set. Then
		# iterate the second asking whether that node has been seen before.
		seen = set()
		na = a
		while na is not None:
			seen.add(na)
			na = na.next
		nb = b
		while nb is not None:
			if nb in seen: return nb
			nb = nb.next
		# I prefer this solution, because the code is shorter and easier to understand,
		# and memory is cheap. Though the O(1) solution is clever if you need it.

	else:
		# O(1) space solution: iterate all the way to the end of the lists. If last
		# nodes match, then hooray, there's an intersection. Keep track of lengths as
		# you do this so you can set one pointer running ahead of the other so they
		# both have the same distance left to run, because the lists could be different
		# lengths. Then iterate the two together, checking for when they become the
		# same thing.
		na = a
		la = 1
		while na.next is not None:
			la += 1
			na = na.next
		nb = b
		lb = 1
		while nb.next is not None:
			lb += 1
			nb = nb.next
		
		if na != nb: return None
		na = a
		nb = b
		while la > lb:
			na = na.next
			la -= 1
		while lb > la:
			nb = nb.next
			lb -= 1

		while na != nb:
			na = na.next
			nb = nb.next
		return na

for o_1 in [False, True]:
	a = Node(1, Node(2, Node(3, Node(4))))
	b = Node(10, Node(11, a.next))
	assert str(seven(a, b, o_1)) == "2->3->4->"
	b.next.next = Node(12)
	assert seven(a, b, o_1) is None

def eight(head):
	tortoise = head
	hare = head.next

	while tortoise != hare and hare is not None:
		tortoise = tortoise.next
		hare = hare.next
		if hare is not None: hare = hare.next

	if hare is None: return None # No loop, we're done

	# Mathematical trickery: When the two meet, tortoise will have stepped x times,
	# and hare will have stepped 2x times. Say the tortoise enters the loop at step k,
	# then at that moment the hare will be at k + (k % l), where l is the size of the loop.
	# At this moment the hare is k%l steps ahead of the tortoise in the loop, which means it
	# is equivalently l - k%l steps behind the tortoise and will catch it in that many
	# additional steps. So x = k + l - (k % l). This I don't know how to solve analytically.
	# Not sure it can be. But think about what it means: We took k steps to get to the loop,
	# then l - k%l steps in to the loop. This is equivalent to saying we're k%l steps from
	# completing the loop, which is the same as k steps if we're willing to potentially go
	# around multiple times. k is also the distance from the beginning of the list to the
	# start of the loop, so if we put a pointer back there and iterate forward, then when
	# the two meet, it will be at the loop's beginning.

	hare = head
	tortoise = tortoise.next # hare moves, tortoise moves too, otherwise they're off by one and never meet
	while hare != tortoise:
		tortoise = tortoise.next
		hare = hare.next
	return tortoise.val

ll = Node(1, Node(2, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9)))))))))
assert eight(ll) is None
ll.next.next.next.next.next.next.next.next.next = ll.next.next.next
# print(ll) runs forever now, but basically 9 points back to 4
assert eight(ll) == 4
