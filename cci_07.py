import random

### One
class Deck:
	def __init__(self):
		self.cards = []
		for suit in ['H', 'D', 'S', 'C']:
			self.cards.append('A' + suit)
			for i in range(2,11):
				self.cards.append(str(i) + suit)
			self.cards.append('J' + suit)
			self.cards.append('Q' + suit)
			self.cards.append('K' + suit)

	def shuffle(self):
		random.shuffle(self.cards)

	def __repr__(self):
		return str(self.cards)

	def discard(self):
		return self.cards.pop(0)

	def draw(self, c):
		# could be a check in here to make sure we're not adding a duplicate or unrecognized card.
		self.cards.append(c)

deck = Deck()
s1 = set(deck.cards)
deck.shuffle()
assert set(deck.cards) == s1 # we lose no cards in a shuffle

class Hand(Deck): # has only some cards in it, but you can still discard, shuffle, print, etc.
	def __init__(self, cards=None):
		self.cards = [] if not cards else cards

class BlackJack:
	def __init__(self, deck, n_players):
		self.deck = deck
		hands = [Hand() for i in range(n_players+1)]
		for i in range(2*n_players+2):
			hands[i%(n_players+1)].draw(deck.discard())
		self.dealer_hand = hands[0]
		self.player_hands = hands[1:]

	def score_hand(self, hand):
		s = [0] # scores possible
		for card in hand.cards:
			if 'A' in card:
				v = 1
			elif card[0] in ['J', 'Q', 'K']:
				v = 10
			else:
				v = int(card[:-1])

			for i in range(len(s)):
				s[i] += v

			if 'A' in card:
				s = s + s
				for i in range(len(s)//2, len(s)):
					s[i] += 10

		# Binary search for 21. If not there, the score is the one closest below. If not there, bust.
		lo = 0
		hi = len(s)-1
		while hi >= lo:
			mid = (lo + hi)//2
			if s[mid] == 21: return 21

			if s[mid] > 21:
				hi = mid - 1
			else:
				lo = mid + 1

		if hi >= 0: return s[hi]
		else: return s[lo]

	## There will also be hit and whatever down here. I don't really know how to play Blackjack well

game = BlackJack(deck, 3)
assert len(game.dealer_hand.cards) == 2
assert len(game.player_hands) == 3
for ph in game.player_hands:
	assert len(ph.cards) == 2
assert len(game.deck.cards) == 52 - 2*4

hand = Hand(['8D', 'AD'])
assert game.score_hand(hand) == 19
hand.cards.append('AS')
assert game.score_hand(hand) == 20

# Gayle has BlackJackHand, even BlackJackCard classes. To me, how you score is part of the *game*,
# not part of a hand, because I consider a hand to be super generic. But this means that in my
# scheme you have to put the logic for scoring and potentially more all in the game. It's a tradeoff
# between having a proliferation of classes and having a potentially unwieldy BlackJack class.

### Two
class CallCenter:
	def __init__(self, employees):
		self.employees = employees
		self.qs = [[],[],[]] # queues for each level

	def dispatchCall(self, call): # give to employee, or put in queue
		if type(call) == str:
			call = Call(number)

		for e in self.employees[call.rank]:
			if not e.ocupado:
				e.ocupado = True
				call.handler = e
				break
		else:
			self.qs[call.rank].append(call)

class Call:
	def __init__(self, number):
		self.number = number
		self.rank = 0 # starts out low level
		self.handler = None # who is taking care of it

class Employee:
	def __init__(self, rank, center):
		self.rank = rank # let's say 0 = respondent, 1 = manager, 2 = director
		self.ocupado = False
		self.call_center = center

	def free(self): # Take up new call
		if len(self.call_center.qs[self.rank]) > 0:
			call = self.call_center.qs[self.rank].pop(0)
			call.handler = self
			self.ocupado = True

	def escalate(self, call):
		call.rank += 1
		self.call_center.dispatchCall(call)

# I'm not gonna run a whole simulation. The point is just to make some sensible classes with
# sensible fields and methods that allow sensible operations.

# Three
# Let's say it's meant to represent a physical jukebox that plays records and takes dollars
# We'll need Record and Money classes, We'll probably need to represent a Record as having a
# couple Songs. Songs are organized by Genre, Artist, year, etc. We might want objects or at
# least enums for many of these. The overall Jukebox will need to be able to search songs by
# features, take payment, maybe play randomly if no one is making a selection. It will need to
# have a way to communicate information out to users. If we want to get really fancy, users
# might have Accounts linked to particular Favorites or Playlists, and we'd need a way to take
# the requisite information to build those. The Jukebox has a Stage (or something) where the
# currently-playing piece of physical media is located. Other media can't be played until it's
# swapped out to be the staged thing.

# Four
# We'll have Spaces, which can be normal or handicapped, superwide or normal or compact, and
# can be full or empty. We might arrange them in to lettered/numbered rows/cols in some cases,
# but not in others. The ParkingLot collects a bunch of Spaces together and might keep track of
# how long cars have been in the lot. If it's paid parking, there might be some form of Ticket
# that parkers generate and put on their dash, and the ParkingLot will know about these and keep
# track of their countdowns if appropriate. We might be able to ask the ParkingLot how many free
# Spaces there are in given rows or levels. If we're really futuristic, then maybe the
# ParkingLot parks Cars all by itself! There might be several types of vehicle, which fit in
# different types of Spaces.

# Five
# We'll need Books, of course. Likely Accounts for users, so we know who has access to which
# Books. We'll need BookMarks for each user associated with their Books, so we can keep their
# places for them. We might want Annotations or Highlights likewise associated betwen Account
# and Book. Accounts might have payment information, like a CreditCard. Books will have scrolling
# and page flip functions. The books will be kept in a Library, which is searchable by Author,
# Genre, year, etc. Accounts can be kept in a list or dictionary, but if managing that gets too
# involved, we might consider a Manager class. On the frontend, the Display itself may have
# Buttons and a View, which get attached to actions (likely JavaScript).

# Six
class Puzzle:
	def __init__(self, N):
		self.N = N

		# generate pieces themselves
		self.pieces = []
		for i in range(N):
			self.pieces.append([])
			for j in range(N):
				self.pieces[i].append(Piece(i*N + j))

		# assign proper matching numbers between their edges, around puzzle's edge is 0
		c = 1
		for i in range(N-1):
			for j in range(N-1):
				self.pieces[i+1][j+1].u = c
				self.pieces[i][j+1].d = c
				self.pieces[i+1][j+1].l = c+1
				self.pieces[i+1][j].r = c+1
				c += 2

			self.pieces[0][i].r = c
			self.pieces[0][i+1].l = c
			self.pieces[i][0].d = c+1
			self.pieces[i+1][0].u = c+1
			c += 2

		self.pieces = [p for row in self.pieces for p in row]
		random.shuffle(self.pieces)

	def solve(self): # only using edges!
		self.grid = [[None]*self.N for i in range(self.N)]

		# find a corner and put it at upper left
		for i,p in enumerate(self.pieces):
			z = sum([c==0 for c in p.sides])
			if z == 2:
				self.grid[0][0] = p
				self.pieces.pop(i)
				break

		# Rotate it until correct.
		while self.grid[0][0].u != 0 or self.grid[0][0].l != 0:
			self.grid[0][0].rotate()

		# finding correct next pieces, down the left side
		for i in range(1,self.N):
			k = 0
			while True:
				y = k//4

				if self.pieces[y].fits(self.grid[i-1][0], 'd'):
					self.grid[i][0] = self.pieces[y]
					self.pieces.pop(y)
					break

				self.pieces[y].rotate()
				k += 1

		# now do the same for all remaining across the rows
		for i in range(self.N):
			for j in range(1,self.N):
				k = 0
				while True:
					y = k//4

					if self.pieces[y].fits(self.grid[i][j-1], 'r'):
						self.grid[i][j] = self.pieces[y]
						self.pieces.pop(y)
						break

					self.pieces[y].rotate()
					k += 1

		return self.grid

class Piece:
	def __init__(self, iden, r=0, u=0, l=0, d=0):
		self.iden = iden # so I can check correctness easily at end
		self.sides = [r, u, l, d]
		self.orientation = random.randint(0, 3) # piece starts out at random rotation
		# orientation 0:  u         orientation 1:  r
		#               l   r                     u   d
		#                 d                         l
		# orientation 2:  d         orientation 3:  l
		#               r   l                     d   u
		#                 u                         r
		# So to get the currently-right side it's [r, d, l, u] across the orientations == sides[-orientation]
		# to get upper side it's [u, r, d, l] == sides[1 - orientation]
		# to get left it's [l, u, r, d] == sides[2 - orientation]
		# and to get down it's [d, l, u, r] == sides[3 - orientation]

	def __repr__(self):
		return str(self.iden)

	@property
	def r(self): # current right edge
		return self.sides[-self.orientation]
	@r.setter
	def r(self, v):
		self.sides[-self.orientation] = v

	@property
	def u(self): # current up edge
		return self.sides[1-self.orientation]
	@u.setter
	def u(self, v):
		self.sides[1-self.orientation] = v

	@property
	def l(self): # current left edge
		return self.sides[2-self.orientation]
	@l.setter
	def l(self, v):
		self.sides[2-self.orientation] = v

	@property
	def d(self): # current down edge
		return self.sides[3-self.orientation]
	@d.setter
	def d(self, v):
		self.sides[3-self.orientation] = v

	def fits(self, other, side): # self goes to the specified `side` of other
		"""Pieces are considered to fit if the unique identifiers across their two edges match"""
		if side == 'r': # self's left has to match other's right
			return other.r == self.l and other.r != 0
		elif side == 'u': # self's down has to match other's up
			return other.u == self.d and other.u != 0
		elif side == 'l': # self't right has to match other's left
			return other.l == self.r and other.l != 0
		elif side == 'd': # self's up has to match other's down
			return other.d == self.u and other.d != 0
		return False

	def rotate(self):
		self.orientation += 1
		self.orientation %= 4

N = 10
puzzle = Puzzle(N)
solution = puzzle.solve()

from cci_01 import seven # my in-place matrix rotation function

sol_upright = str([[i*N + j for j in range(N)] for i in range(N)])
for i in range(4):
	if str(solution) == sol_upright:
		break
	seven(solution)
else:
	assert False # We have to have hit the break statement, or we'll hit this.

# Seven
# We'll need a database holding Conversations. Conversations might contain Messages, Images, or
# other Media. Accounts exist, with associations to other accounts and to particular Conversations.
# If a new Message is sent in a Conversation, all the Accounts associated to it get a Notification.
# The greatest challenge of this is scale. Conversations might run for years and have a *lot* of
# associated information. We'll need to store across multiple machines and fetch it by date range
# when a user scrolls to a particular point. Conversations can be created by Accounts, and other
# users can be added by their Accounts. You might in some cases have guardrails, like I can't start
# a conversation with you if we're not already contacts, but I prefer the raw model, like Reddit.
# An Account might have an online field, which allows other users to tell whether they're active.
# We might have a Manager for Accounts, to hold all the logic related to creating, deleting, and
# changing their state.

# Eight
class Othello: # https://www.othelloonline.org/
	def __init__(self):
		self.board = [['_']*8 for i in range(8)]
		self.board[3][3] = 'w'
		self.board[4][4] = 'w'
		self.board[3][4] = 'b'
		self.board[4][3] = 'b'
		self.whose_move = 'b'

	def move(self, i, j, just_check_legality=False):
		if self.board[i][j] != '_': return False
		flag = False

		for v in [-1, 0, 1]:
			for h in [-1, 0, 1]:
				y = i
				x = j
				k = 0
				while True:
					y += v
					x += h

					if not 0 <= y < 8 or not 0 <= x < 8 or self.board[y][x] == '_':
						break
					if self.board[y][x] == self.whose_move:
						for k in range(k, 0, -1): # iterate back along the ray
							y -= v
							x -= h
							flag = True # something got changed, so it's a legal move
							if not just_check_legality: # then actually modify the board
								self.board[y][x] = 'b' if self.whose_move == 'b' else 'w' # flipping chips
							else: break
						break

					k += 1 # increments if we find an opposite-colored piece

		if flag and not just_check_legality:
			self.board[i][j] = self.whose_move
			self.whose_move = 'w' if self.whose_move == 'b' else 'b'
		return flag

	def check_moves(self):
		for i in range(8):
			for j in range(8):
				if self.move(i, j, just_check_legality=True):
					return True
		self.whose_move = 'w' if self.whose_move == 'b' else 'b'
		return False

	def score(self):
		b = 0
		w = 0
		for i in range(8):
			for j in range(8):
				if self.board[i][j] == 'b':
					b += 1
				elif self.board[i][j] == 'w':
					w += 1
		return b,w

	def __repr__(self):
		return '\n'.join(str(row) for row in self.board)

game = Othello()
assert str(game) == """['_', '_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', 'w', 'b', '_', '_', '_']
['_', '_', '_', 'b', 'w', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_', '_']"""

# You'd hide this behind a GUI, but this is what would be going on: You move, check whether
# the opponent can move, if so, they move, if not you check whether you can move, and when
# neither can move, the game ends. There'd be a while loop and a little extra logic. Here I'm
# recapitulating a game I played against a computer just to make sure all the internal states
# are right throughout the process.
for move in [(2,3), (4,2), (5,5), (2,5), (4,1), (5,2), (2,4), (3,0), (4,0), (5,0), (5,3), (5,4),
	(6,5), (2,2), (1,1), (3,2), (6,2), (4,5), (1,4), (0,3), (2,6), (7,5), (6,6), (7,2), (5,1),
	(5,7), (7,7), (3,5), (4,6), (4,7), (7,6), (0,0), (7,4), (2,7), (3,7), (6,0), (6,7), (2,1),
	(1,7), (6,3), (2,0), (1,0), (7,3), (1,5), (0,5), (0,4), (0,2), (0,6), (0,7), (1,3), (1,2),
	(3,1), (7,1), (6,4), (6,1), (3,6), (1,6)]:
	assert game.move(*move)
	assert game.check_moves() # whether the opponent can now move. Flips move back to same player if False.
for move in [(0,1), (7,0), (5,6)]:
	assert game.move(*move)
	assert not game.check_moves() # whether the oponent can now move
assert not game.check_moves() # When the game truly ends, neither has a move left.

assert str(game) == """['w', 'w', 'b', 'b', 'b', 'b', 'b', 'b']
['w', 'w', 'w', 'b', 'b', 'b', 'b', 'b']
['w', 'w', 'w', 'w', 'w', 'b', 'b', 'b']
['w', 'w', 'w', 'w', 'w', 'w', 'w', 'b']
['w', 'w', 'w', 'w', 'w', 'w', 'w', 'b']
['w', 'w', 'w', 'w', 'w', 'w', 'w', 'b']
['w', 'w', 'b', 'b', 'w', 'b', 'b', 'b']
['w', 'b', 'b', 'b', 'b', 'b', 'b', 'b']"""
assert game.score() == (29,35)

# Nine
class CircularList:
	def __init__(self, data):
		self.data = data
		self.s = 0
		self.n = len(self.data)

	def rotate(self, x):
		self.s += x
		self.s %= self.n

	def __iter__(self): # making something iterable in python entails implementing __iter__ and __next__
		self.i = 0
		return self

	def __next__(self):
		if self.i < len(self.data):
			r = self.data[(self.s + self.i) % self.n]
			self.i += 1
			return r
		else:
			raise StopIteration

nine = CircularList([0,1,2,3,4,5,6,7,8,9])
nine.rotate(4)
assert [a for a in nine] == [4,5,6,7,8,9,0,1,2,3] # nine isn't a list, but it is iterable, so make list to compare

# Ten
class Minesweeper:
	def __init__(self, N, B):
		self.N = N
		self.board = [['_']*N for i in range(N)]

		self.la_bomba = [1]*B + [0]*(N*N - B)
		random.shuffle(self.la_bomba) # shuffle ensures minimal random calls
		self.la_bomba = set([divmod(x, N) for x,b in enumerate(self.la_bomba) if b == 1])

		self.n_bombs_around = [[0]*N for i in range(N)] # precompute the numbers for each square
		for i,j in self.la_bomba:
			for v in [-1, 0, 1]: # vertical offset
				for h in [-1, 0, 1]: # horizontal offset
					self.n_bombs_around[i+v][j+h] += 1

		self.over = False

	def __repr__(self):
		return '\n'.join(str(row) for row in self.board)

	def flag(self, i, j):
		self.board[i][j] = 'f'

	def click(self, i, j):
		if self.over: return

		bueno = self._recurse(i, j)
		if not bueno: # show all the bombs
			for r,c in self.la_bomba:
				self.board[r][c] = '*'
			self.over = True
			return "lose" # could set some kind of game state instead

		else: # check whether there's a not-bomb locaton that still hasn't been explored
			for r in range(self.N):
				for c in range(self.N):
					if self.board[r][c] == '_' and (r,c) not in self.la_bomba:
						return # if one is found
			self.over = True
			return "win" # if none found

	def _recurse(self, i, j):
		"""True if you live, False if you die"""
		if (i,j) in self.la_bomba: return False
		if self.board[i][j] != '_': return True # already explored (or flagged)
					
		if self.n_bombs_around[i][j] > 0:
			self.board[i][j] = str(self.n_bombs_around[i][j])
		else:
			self.board[i][j] = ' '
			for h in [-1, 0, 1]: # horizontal offset
				for v in [-1, 0, 1]: # vertical offset
					y = i+v
					x = j+h
					if 0 <= y < self.N and 0 <= x < self.N:
						self._recurse(y, x)
		return True

random.seed(11)
game = Minesweeper(7, 3)
assert str(game) == """['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']"""
game.click(0, 0)
assert str(game) == """[' ', ' ', ' ', ' ', ' ', ' ', ' ']
[' ', ' ', ' ', ' ', '1', '1', '1']
['1', '1', '1', ' ', '1', '_', '_']
['_', '_', '1', '1', '2', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']"""
game.click(3, 0)
assert str(game) == """[' ', ' ', ' ', ' ', ' ', ' ', ' ']
[' ', ' ', ' ', ' ', '1', '1', '1']
['1', '1', '1', ' ', '1', '_', '_']
['1', '_', '1', '1', '2', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']"""
game.flag(3,1)
game.flag(2,5)
assert str(game) == """[' ', ' ', ' ', ' ', ' ', ' ', ' ']
[' ', ' ', ' ', ' ', '1', '1', '1']
['1', '1', '1', ' ', '1', 'f', '_']
['1', 'f', '1', '1', '2', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']
['_', '_', '_', '_', '_', '_', '_']"""
game.click(6,6)
assert str(game) == """[' ', ' ', ' ', ' ', ' ', ' ', ' ']
[' ', ' ', ' ', ' ', '1', '1', '1']
['1', '1', '1', ' ', '1', 'f', '_']
['1', 'f', '1', '1', '2', '2', '1']
['1', '1', '1', '1', '_', '1', ' ']
[' ', ' ', ' ', '1', '1', '1', ' ']
[' ', ' ', ' ', ' ', ' ', ' ', ' ']"""
from copy import deepcopy
losing_game = deepcopy(game)
assert losing_game.click(4,4) == 'lose'
assert losing_game.over
assert str(losing_game) == """[' ', ' ', ' ', ' ', ' ', ' ', ' ']
[' ', ' ', ' ', ' ', '1', '1', '1']
['1', '1', '1', ' ', '1', '*', '_']
['1', '*', '1', '1', '2', '2', '1']
['1', '1', '1', '1', '*', '1', ' ']
[' ', ' ', ' ', '1', '1', '1', ' ']
[' ', ' ', ' ', ' ', ' ', ' ', ' ']"""
assert not game.over
assert game.click(2,6) == 'win'
assert game.over
assert str(game) == """[' ', ' ', ' ', ' ', ' ', ' ', ' ']
[' ', ' ', ' ', ' ', '1', '1', '1']
['1', '1', '1', ' ', '1', 'f', '1']
['1', 'f', '1', '1', '2', '2', '1']
['1', '1', '1', '1', '_', '1', ' ']
[' ', ' ', ' ', '1', '1', '1', ' ']
[' ', ' ', ' ', ' ', ' ', ' ', ' ']"""

# Gayle uses all kinds of classes for this, a Cell which knows its value and whether it is exposed, 
# a Board separate from the Game, and a couple other tiny things. I think it can work well as a
# single class, but for more complicated things, sure, make classes, put complexity in classes.

# Eleven
# I'd probably keep a tree of directories and files. Each Node can have many children or contain an
# actual file. You could optionally make Directory and File classes which inherit from Node. Nodes
# know who their parent is. Nodes support operations like deleting themselves, moving themselves,
# or adding themselves at a particular location, given a ref to the tree. Nodes keep track of metadata
# like how large the file or sum of things below is, when they were created, when files (leaf nodes)
# were last updated, their name, etc.
# Not gonna code it right now. Working with trees is too familiar, wouldn't add anything.

# Twelve
from cci_02 import Node

class HashTable:
	def __init__(self, n=10, p=0.5):
		self.arr = [None]*n
		self.n = n # the length of the backing table
		self.p = p # the portion of things that can be in the table before a resize
		self.c = 0 # how many things are in the table and its lists

	def __getitem__(self, k):
		node, parent, i = self._get_node_and_parent(k)

		if node is None: raise KeyError # if there is no such key in the data structure

		return node.val[1]

	def __setitem__(self, k, v):
		self.c += 1 # there is now one more item in the hashtable

		if float(self.c) / self.n > self.p:
			self._double()

		node, parent, i = self._get_node_and_parent(k)
		
		if node is None and parent is None: # No linked list head at this array location
			self.arr[i] = Node((k,v))
		elif node is not None: # an entry with this very key already exists, so just update
			node.val = (k,v)
		else: # node is None, but parent isn't, so chain new node to end of list
			parent.next = Node((k,v))

	def __delitem__(self, k):
		node, parent, i = self._get_node_and_parent(k)

		if node is None: raise KeyError

		if parent is None: # move the head of the list forward
			self.arr[i] = node.next
		else: # skip node
			parent.next = node.next

		self.c -= 1 # if we didn't keyerror, there is now one fewer items in the hashtable

	def _double(self):
		old_arr = self.arr
		self.n *= 2
		self.arr = [None]*self.n
		self.c = 1 # I'm about to call setitem a bunch, so all the previously-added items are about to get recounted

		for node in old_arr: # might be None or a node
			while node is not None: # loop along linked list's length
				self.__setitem__(*node.val) # puts (k,v) in to new, longer self.arr
				node = node.next

	def _get_node_and_parent(self, k):
		"""for getting and setting and deleting, I may need three pieces of information: the table index
		where a node with the key should go, the node already associated to a key if it exists, and that
		node's parent"""
		i = hash(k) % self.n

		if self.arr[i] is None: return None, None, i

		parent = None
		node = self.arr[i]
		while node is not None and node.val[0] != k:
			parent = node
			node = node.next # Can run off the end, in which case parent is the lat node in the linked list

		return node, parent, i

	def __repr__(self):
		return '\n'.join(str(i) + ' ' + str(node) for i,node in enumerate(self.arr))

	def __len__(self):
		return self.c

eleven = HashTable(n=10, p=0.5)
# Here I'm exploiting the fact that hash(int) = int and my knowledge of how large the array should be to
# put things exactly where I expect.
eleven[0] = "zero"
eleven[2] = "two"
eleven[10] = "ten"
eleven[20] = "twenty"
eleven[7] = "seven"
assert str(eleven) == """0 (0, 'zero')->(10, 'ten')->(20, 'twenty')->
1 None
2 (2, 'two')->
3 None
4 None
5 None
6 None
7 (7, 'seven')->
8 None
9 None"""
assert eleven[7] == 'seven' # the just-head linked list case
assert eleven[0] == 'zero' # front
assert eleven[10] == 'ten'
assert eleven[20] == 'twenty'
try:
	eleven[5]
	assert False
except KeyError:
	assert True
assert len(eleven) == 5
del eleven[10]
assert len(eleven) == 4
assert str(eleven) == """0 (0, 'zero')->(20, 'twenty')->
1 None
2 (2, 'two')->
3 None
4 None
5 None
6 None
7 (7, 'seven')->
8 None
9 None"""
del eleven[0]
del eleven[7]
assert str(eleven) == """0 (20, 'twenty')->
1 None
2 (2, 'two')->
3 None
4 None
5 None
6 None
7 None
8 None
9 None"""
try:
	del eleven[7] # the table is None here
	assert False
except KeyError:
	assert True
try:
	del eleven[0] # the table has the 20 node here, but key 0 doesn't exist
	assert False
except KeyError:
	assert True
eleven[0] = "zero"
eleven[6] = "six"
eleven[1010] = "one thousand ten"
assert str(eleven) == """0 (20, 'twenty')->(0, 'zero')->(1010, 'one thousand ten')->
1 None
2 (2, 'two')->
3 None
4 None
5 None
6 (6, 'six')->
7 None
8 None
9 None"""
eleven[18] = "eighteen" # check doubling and re-populating works properly
assert str(eleven) == """0 (20, 'twenty')->(0, 'zero')->
1 None
2 (2, 'two')->
3 None
4 None
5 None
6 (6, 'six')->
7 None
8 None
9 None
10 (1010, 'one thousand ten')->
11 None
12 None
13 None
14 None
15 None
16 None
17 None
18 (18, 'eighteen')->
19 None"""
assert len(eleven) == 6
