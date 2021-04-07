
### One
import random

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

	## Then there will be hit, whatever down here. I don't really know how to play Blackjack well

game = BlackJack(deck, 3)
assert len(game.dealer_hand.cards) == 2
assert len(game.player_hands) == 3
for ph in game.player_hands:
	assert len(ph.cards) == 2
assert len(game.deck.cards) == 52 - 2*4

### Two


