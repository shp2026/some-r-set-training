"""
somerset_tricks.py
Trick-playing and legal-move logic for Some-R-Set.

Depends on somerset_deck.py being in the same directory.

Key rules encoded here:
  1. Leading:
       - If trump was named, first card of the FIRST trick must be trump.
       - Subsequent tricks (and no-trump hands) may lead any card.

  2. Following (given a led suit):
       - Must follow led suit IF you have one, UNLESS you play:
           a) the 0/0 card  (always legal)
           b) a trump card  (always legal when trump was named)
       - If you have no card of the led suit, play anything.

  3. Trick winner:
       Priority 1 — Any trump card was played  → highest trump value wins.
                    (S/S is the lowest trump, so it loses to any suited trump.)
       Priority 2 — No trump, but 0/0 was played → highest "double" card wins.
                    Doubles are: 0/0, 2/2, 4/4, 6/6, 8/8, 10/10, 12/12.
                    0/0 is the lowest double.
       Priority 3 — No trump, no 0/0 → highest value in the led suit wins.
"""

from somerset_deck import (
    Card, MASTER_DECK, SUIT_DEFS, SS_SUIT,
    SUIT_VALUE_TO_CARD, INDEX_TO_CARD,
)


# ── Constants derived from the deck ──────────────────────────────────────────

# The 0/0 card (Zeros suit, value 0, index 0)
CARD_0_0: Card = SUIT_VALUE_TO_CARD[("Zeros", 0)]

# All "double" cards: the highest-value card of each regular suit
# (the card whose value == suit max, i.e. 0/0, 2/2, 4/4 … 12/12)
# Sorted by value ascending so 0/0 is lowest double.
DOUBLE_CARDS: list[Card] = sorted(
    [SUIT_VALUE_TO_CARD[(suit, max_val)] for suit, max_val in SUIT_DEFS],
    key=lambda c: c.value,
)
DOUBLE_INDICES: frozenset[int] = frozenset(c.index for c in DOUBLE_CARDS)

# S/S wild card
CARD_SS: Card = INDEX_TO_CARD[49]


# ── Legal move logic ──────────────────────────────────────────────────────────

def legal_plays(
    hand:        list[Card],
    led_card:    Card | None,
    trump_suit:  str  | None,
    is_first_trick: bool = False,
    is_leading:  bool = False,
) -> list[Card]:
    """
    Return the subset of `hand` that may legally be played.

    Parameters
    ----------
    hand           : cards held by the current player
    led_card       : the card that opened this trick (None if this player leads)
    trump_suit     : the named trump suit, or None for no-trump
    is_first_trick : True only for the very first trick of the hand
    is_leading     : True when this player is the one leading the trick
    """
    if not hand:
        return []

    # ── Leading ──────────────────────────────────────────────────────
    if is_leading or led_card is None:
        if is_first_trick and trump_suit is not None:
            # Must lead a trump card to open the hand
            trump_cards = [c for c in hand if _is_trump(c, trump_suit)]
            # (Should always be non-empty if rules are followed, but fall back
            # to any card defensively.)
            return trump_cards if trump_cards else hand[:]
        return hand[:]   # any card is fine otherwise

    # ── Following ─────────────────────────────────────────────────────
    led_suit = _effective_suit(led_card, trump_suit)
    same_suit = [c for c in hand if _effective_suit(c, trump_suit) == led_suit]

    if not same_suit:
        # No matching suit — free to play anything
        return hand[:]

    # Has matching suit: must follow, BUT 0/0 and trump are always allowed
    always_legal = []
    if CARD_0_0 in hand:
        always_legal.append(CARD_0_0)
    if trump_suit is not None:
        always_legal += [c for c in hand if _is_trump(c, trump_suit)
                         and c not in always_legal]

    # Combine, deduplicate, preserve order
    legal = list(same_suit)
    for c in always_legal:
        if c not in legal:
            legal.append(c)
    return legal


# ── Trick winner logic ────────────────────────────────────────────────────────

def trick_winner(
    trick:       list[tuple[Card, int]],   # (card, player_index) in play order
    trump_suit:  str | None,
) -> tuple[Card, int]:
    """
    Determine which (card, player) wins the trick.

    Parameters
    ----------
    trick      : list of (card, player_index) in the order they were played.
                 trick[0] is the led card.
    trump_suit : named trump suit, or None.

    Returns
    -------
    (winning_card, winning_player_index)
    """
    if not trick:
        raise ValueError("trick is empty")

    led_card = trick[0][0]

    # ── Priority 1: trump ─────────────────────────────────────────────
    trump_plays = [(c, p) for c, p in trick
                   if trump_suit and _is_trump(c, trump_suit)]
    if trump_plays:
        return max(trump_plays, key=lambda cp: _trump_rank(cp[0], trump_suit))

    # ── Priority 2: 0/0 was led or played → doubles hierarchy ─────────
    played_cards = [c for c, _ in trick]
    if CARD_0_0 in played_cards:
        double_plays = [(c, p) for c, p in trick if c.index in DOUBLE_INDICES]
        if double_plays:
            return max(double_plays, key=lambda cp: cp[0].value)
        # 0/0 was played but no other doubles — 0/0 wins by default
        return next((c, p) for c, p in trick if c == CARD_0_0)

    # ── Priority 3: highest card in led suit ───────────────────────────
    led_suit = led_card.suit
    led_suit_plays = [(c, p) for c, p in trick if c.suit == led_suit]
    return max(led_suit_plays, key=lambda cp: cp[0].value)


# ── Trump-rank helper ─────────────────────────────────────────────────────────

def _trump_rank(card: Card, trump_suit: str) -> int:
    """
    Numeric rank for comparing trump cards.
    S/S = -1 (lowest trump); suited trump cards rank by their face value.
    """
    if card.suit == SS_SUIT:
        return -1          # S/S is always the lowest trump
    return card.value


def _is_trump(card: Card, trump_suit: str | None) -> bool:
    """True if the card belongs to the trump suit or is the S/S wild."""
    if card.suit == SS_SUIT:
        return True
    if trump_suit is None:
        return False
    return card.suit == trump_suit


def _effective_suit(card: Card, trump_suit: str | None) -> str:
    """
    The suit a card 'counts as' for following-suit purposes.
    S/S and trump-suited cards all count as trump_suit when trump is named.
    """
    if _is_trump(card, trump_suit):
        return trump_suit
    return card.suit


# ── Trick summary helper ──────────────────────────────────────────────────────

def trick_summary(
    trick:      list[tuple[Card, int]],
    trump_suit: str | None,
) -> dict:
    """
    Return a dict summarising a completed trick — useful for logging/reward.
    """
    winner_card, winner_player = trick_winner(trick, trump_suit)
    bonus = sum(c.bonus_points for c, _ in trick)
    return {
        "winner_player" : winner_player,
        "winner_card"   : winner_card,
        "bonus_points"  : bonus,
        "cards_played"  : [c for c, _ in trick],
        "trump_suit"    : trump_suit,
    }


# ── Tests / demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from somerset_deck import SUIT_VALUE_TO_CARD as SV

    def card(suit, val):
        return SV[(suit, val)]

    def ss():
        return CARD_SS

    def show_trick(label, trick, trump):
        s = trick_summary(trick, trump)
        print(f"\n{label}")
        print(f"  Trump : {trump or 'none'}")
        for c, p in trick:
            print(f"    Player {p}: {c}")
        print(f"  → Winner: Player {s['winner_player']} "
              f"with {s['winner_card']}  (bonus: +{s['bonus_points']} pts)")

    # 1. Trump wins over led suit
    show_trick(
        "1. Trump card beats led suit",
        trick=[
            (card("Eights", 8), 0),   # leads 8/8 (highest Eights)
            (card("Eights", 3), 1),   # follows suit
            (card("Tens",   2), 2),   # plays trump (Tens is trump)
            (card("Eights", 5), 3),
        ],
        trump="Tens",
    )

    # 2. S/S (lowest trump) still beats all non-trump
    show_trick(
        "2. S/S beats non-trump even though it's lowest trump",
        trick=[
            (card("Twelves", 12), 0),  # leads highest Twelves
            (card("Twelves",  9), 1),
            (ss(),                2),  # plays S/S (trump = Sixes)
            (card("Twelves",  7), 3),
        ],
        trump="Sixes",
    )

    # 3. 0/0 triggers doubles: highest double wins
    show_trick(
        "3. 0/0 played — doubles hierarchy takes over",
        trick=[
            (card("Sixes",  6), 0),   # leads 6/6 (a double!)
            (card("Sixes",  4), 1),
            (CARD_0_0,          2),   # plays 0/0 — activates doubles rule
            (card("Tens",  10), 3),   # 10/10 is the highest double here
        ],
        trump=None,
    )

    # 4. 0/0 played but no other doubles — 0/0 itself wins
    show_trick(
        "4. 0/0 played, no other doubles — 0/0 wins",
        trick=[
            (card("Sixes",  5), 0),
            (card("Sixes",  4), 1),
            (CARD_0_0,          2),
            (card("Sixes",  3), 3),
        ],
        trump=None,
    )

    # 5. No trump, no 0/0 — highest in led suit wins
    show_trick(
        "5. Plain trick — highest in led suit wins",
        trick=[
            (card("Fours",  1), 0),
            (card("Fours",  4), 1),   # highest Fours
            (card("Sixes",  6), 2),   # different suit, irrelevant
            (card("Fours",  2), 3),
        ],
        trump=None,
    )

    # 6. Legal moves — must follow suit but may play 0/0 or trump instead
    print("\n6. Legal plays — hand has Eights and trump (Tens), led suit is Eights")
    hand = [
        card("Eights", 3),
        card("Eights", 7),
        card("Tens",   4),   # trump
        card("Sixes",  5),
        CARD_0_0,            # always legal
    ]
    led = card("Eights", 1)
    legal = legal_plays(hand, led_card=led, trump_suit="Tens")
    print(f"  Trump    : Tens")
    print(f"  Led card : {led}")
    print(f"  Hand     : {[str(c) for c in hand]}")
    print(f"  Legal    : {[str(c) for c in legal]}")
