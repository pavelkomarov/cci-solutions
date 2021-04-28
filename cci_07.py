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
		grid = [[None]*self.N for i in range(self.N)]

		# find a corner and put it at upper left
		for i,p in enumerate(self.pieces):
			z = sum([c==0 for c in p.sides])
			if z == 2:
				grid[0][0] = p
				self.pieces.pop(i)
				break

		# Rotate it until correct.
		while grid[0][0].u != 0 or grid[0][0].l != 0:
			grid[0][0].rotate()

		# finding correct next pieces, down the left side
		for i in range(1,self.N):
			k = 0
			while True:
				y = k//4

				if self.pieces[y].fits(grid[i-1][0], 'd'):
					grid[i][0] = self.pieces[y]
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

					if self.pieces[y].fits(grid[i][j-1], 'r'):
						grid[i][j] = self.pieces[y]
						self.pieces.pop(y)
						break

					self.pieces[y].rotate()
					k += 1

		return grid

class Piece:
	def __init__(self, iden, r=0, u=0, l=0, d=0):
		self.iden = iden # so I can check correctness easily at end
		self.sides = [r, u, l, d]
		self.orientation = random.randint(0, 3)
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
	def r(self): # currently right edge
		return self.sides[-self.orientation]
	@r.setter
	def r(self, v):
		self.sides[-self.orientation] = v

	@property
	def u(self): # currently up edge
		return self.sides[1-self.orientation]
	@u.setter
	def u(self, v):
		self.sides[1-self.orientation] = v

	@property
	def l(self): # currently left edge
		return self.sides[2-self.orientation]
	@l.setter
	def l(self, v):
		self.sides[2-self.orientation] = v

	@property
	def d(self): # currently down edge
		return self.sides[3-self.orientation]
	@d.setter
	def d(self, v):
		self.sides[3-self.orientation] = v

	def fits(self, other, side): # self goes to the specified `side` of other
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

# Ten


