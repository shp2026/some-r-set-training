"""
somerset_deck.py
Card deck structure for Some-R-Set.

Suit structure:
  Suit name  | Max value | Card count
  -----------+-----------+-----------
  Zeros      |     0     |     1
  Twos       |     2     |     3
  Fours      |     4     |     5
  Sixes      |     6     |     7
  Eights     |     8     |     9
  Tens       |    10     |    11
  Twelves    |    12     |    13
  S/S        |    -1     |     1   (lowest trump, unique)
  -----------+-----------+-----------
  Total                       50

Bonus point cards (added to trick value when won):
  +3 pts : S/S
  +1 pt  : 1/2  (Twos-1),   2/4  (Fours-2),  3/6  (Sixes-3)
  +2 pts : 4/8  (Eights-4), 5/10 (Tens-5),   6/12 (Twelves-6)

Card naming convention: "V/S" where V = value, S = suit's max value.
e.g. "4/8" = value 4 in the suit of Eights.

Each card is represented as a frozen dataclass for safety.
Cards have a unique integer index (0–49) for use in Gym obs/action vectors.
"""

from dataclasses import dataclass
from typing import Optional
import random


# ── Suit definitions ──────────────────────────────────────────────────────────

# (suit_name, max_value)  →  card count = max_value + 1
SUIT_DEFS = [
    ("Zeros",   0),
    ("Twos",    2),
    ("Fours",   4),
    ("Sixes",   6),
    ("Eights",  8),
    ("Tens",   10),
    ("Twelves",12),
]

SS_SUIT  = "S/S"
SS_VALUE = -1

# Bonus points keyed by (suit_name, value).
# The naming pattern V/S means: value V in the suit whose max is S.
BONUS_POINTS: dict[tuple, int] = {
    (SS_SUIT,SS_VALUE): 3,   # S/S wild card
    ("Twos",        1): 1,   # 1/2
    ("Fours",       2): 1,   # 2/4
    ("Sixes",       3): 1,   # 3/6
    ("Eights",      4): 2,   # 4/8
    ("Tens",        5): 2,   # 5/10
    ("Twelves",     6): 2,   # 6/12
}

TOTAL_BONUS_POINTS = 12


# ── Card dataclass ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Card:
    """
    Immutable representation of a single Some-R-Set card.

    Attributes
    ----------
    index        : unique integer 0-49; used as the Gym action/obs index
    suit         : suit name string  (e.g. "Eights", "S/S")
    value        : integer rank within the suit
    is_wild      : True means can be laid on any trump suit, i.e. the S/S and 0/0 cards
    bonus_points : extra points added to trick value when this card is won
    """
    index        : int
    suit         : str
    value        : Optional[int]
    is_wild      : bool = False
    bonus_points : int  = 0

    @property
    def short_name(self) -> str:
        """
        Canonical V/S notation, e.g. '4/8', '0/0', 'S/S'.
        For regular cards: '{value}/{suit_max}'.
        """
        if self.suit == SS_SUIT:
            return "S/S"
        suit_max = next(mx for nm, mx in SUIT_DEFS if nm == self.suit)
        return f"{self.value}/{suit_max}"

    def __str__(self):
        bp = f"  [+{self.bonus_points} pts]" if self.bonus_points else ""
        if self.is_wild:
            return f"S/S (wild trump){bp}"
        return f"{self.short_name} ({self.suit}){bp}"

    def __repr__(self):
        return (f"Card(index={self.index}, suit={self.suit!r}, "
                f"value={self.value}, bonus={self.bonus_points})")


# ── Build the full 50-card deck ───────────────────────────────────────────────

def _build_deck() -> list[Card]:
    cards = []
    idx = 0

    for suit_name, max_val in SUIT_DEFS:
        for value in range(max_val + 1):
            bonus = BONUS_POINTS.get((suit_name, value), 0)
            cards.append(Card(
                index        = idx,
                suit         = suit_name,
                value        = value,
                bonus_points = bonus,
            ))
            idx += 1

    # S/S wild card — always index 49
    cards.append(Card(
        index        = idx,
        suit         = SS_SUIT,
        value        = SS_VALUE,
        is_wild      = True,
        bonus_points = BONUS_POINTS[(SS_SUIT, SS_VALUE)],
    ))

    assert len(cards) == 50, f"Expected 50 cards, got {len(cards)}"
    assert sum(c.bonus_points for c in cards) == TOTAL_BONUS_POINTS, \
        "Bonus point total mismatch"
    return cards


# Module-level immutable master deck — treat as read-only
MASTER_DECK: list[Card] = _build_deck()

# Fast lookups
INDEX_TO_CARD:      dict[int,   Card] = {c.index: c for c in MASTER_DECK}
SUIT_VALUE_TO_CARD: dict[tuple, Card] = {(c.suit, c.value): c for c in MASTER_DECK}
BONUS_CARD_INDICES: frozenset[int]    = frozenset(
    c.index for c in MASTER_DECK if c.bonus_points > 0
)


# ── Deck utilities ────────────────────────────────────────────────────────────

def new_shuffled_deck(rng: Optional[random.Random] = None) -> list[Card]:
    """Return a fresh shuffled copy of the 50-card deck."""
    deck = list(MASTER_DECK)
    (rng or random).shuffle(deck)
    return deck


def deal(deck: list[Card], num_players: int) -> list[list[Card]]:
    """
    Deal the entire deck round-robin among num_players.
    Returns a list of hands. With 50 cards and 4 players,
    players 0 and 1 receive 13 cards; players 2 and 3 receive 12.
    """
    hands: list[list[Card]] = [[] for _ in range(num_players)]
    for i, card in enumerate(deck):
        hands[i % num_players].append(card)
    return hands


def trick_bonus(trick: list[Card]) -> int:
    """Sum of bonus points carried by cards in a completed trick."""
    return sum(c.bonus_points for c in trick)


def cards_by_suit(cards: list[Card]) -> dict[str, list[Card]]:
    """Group cards by suit, sorted by value within each suit."""
    groups: dict[str, list[Card]] = {}
    for card in cards:
        groups.setdefault(card.suit, []).append(card)
    for suit in groups:
        groups[suit].sort(key=lambda c: (c.value is None, c.value))
    return groups


def suit_order() -> list[str]:
    """Suits in ascending order of max value, S/S last."""
    return [name for name, _ in SUIT_DEFS] + [SS_SUIT]


# ── Trump helpers ─────────────────────────────────────────────────────────────

def trump_cards(trump_suit: str) -> list[Card]:
    """
    All trump cards in ascending trump order (lowest first).
    S/S is always the lowest trump regardless of suit called.
    """
    ss = INDEX_TO_CARD[49]
    suit_trumps = sorted(
        (c for c in MASTER_DECK if c.suit == trump_suit and not c.is_wild),
        key=lambda c: c.value,
    )
    return [ss] + suit_trumps   # S/S = lowest trump


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def hand_str(hand: list[Card]) -> str:
    """Human-readable hand, grouped by suit, bonus cards flagged."""
    groups = cards_by_suit(hand)
    lines = []
    for suit in suit_order():
        if suit not in groups:
            continue
        parts = []
        for c in groups[suit]:
            label = "wild" if c.is_wild else str(c.value)
            if c.bonus_points:
                label += f"(+{c.bonus_points})"
            parts.append(label)
        lines.append(f"  {suit:8s}: {', '.join(parts)}")
    return "\n".join(lines)


# ── Sanity check / demo ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Total cards : {len(MASTER_DECK)}")
    print(f"Bonus cards : {len(BONUS_CARD_INDICES)}  "
          f"(total bonus points in deck: {TOTAL_BONUS_POINTS})")
    print()

    print("Full deck — bonus cards highlighted:")
    for suit_name, cards in cards_by_suit(MASTER_DECK).items():
        for c in cards:
            flag = f"  ← +{c.bonus_points} pts" if c.bonus_points else ""
            print(f"  {c.short_name:6s}  {suit_name:8s}  idx={c.index:2d}{flag}")
    print()

    # 4-player deal
    rng   = random.Random(42)
    deck  = new_shuffled_deck(rng)
    hands = deal(deck, 4)
    print("Sample 4-player deal (seed=42):")
    for i, hand in enumerate(hands):
        bp = sum(c.bonus_points for c in hand)
        print(f"\nPlayer {i} ({len(hand)} cards, {bp} bonus pts in hand):")
        print(hand_str(hand))

    # Trick bonus example
    print("\nExample: player wins a trick containing 4/8 and S/S:")
    trick = [
        SUIT_VALUE_TO_CARD[("Eights", 4)],   # +2 pts
        INDEX_TO_CARD[49],                    # S/S +3 pts
        SUIT_VALUE_TO_CARD[("Tens",    3)],   # +0 pts
        SUIT_VALUE_TO_CARD[("Sixes",   2)],   # +0 pts
    ]
    print(f"  Cards : {', '.join(str(c) for c in trick)}")
    print(f"  Bonus : +{trick_bonus(trick)} pts")
