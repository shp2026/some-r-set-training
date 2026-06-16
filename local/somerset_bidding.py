"""
somerset_bidding.py
Dealing (with kitty) and bidding logic for Some-R-Set.

Dealing:
  - 50 cards dealt 12 to each of 4 players; 2 remainder form the kitty.
  - Kitty is face-up and public; both cards go into the first trick.
  - Whoever wins the first trick collects the kitty cards (and their bonuses).

Bidding:
  - Starts with the player to the left of the dealer (first_bidder).
  - Each player may bid or pass; bids must strictly exceed the current high bid.
  - Minimum bid: MIN_BID (default 6). Enforced to prevent misdeal Nash traps.
  - Bidding ends when three consecutive players pass after a bid is made,
    OR (rarely in practice) all four players pass → misdeal.
  - The bid winner names trump (or declares no-trump) and leads the first trick.
  - Bid winner leads the first trick, so the kitty bonus points are theirs
    to win (or lose to a trump/0-0 play).
"""

from dataclasses import dataclass, field
from typing import Optional
from somerset_deck import Card, new_shuffled_deck, TOTAL_BONUS_POINTS
import random

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_PLAYERS   = 4
CARDS_PER_HAND = 12
KITTY_SIZE    = 2          # 50 - 4×12 = 2
MIN_BID       = 6          # enforced floor; prevents misdeal equilibrium
MAX_BID       = 24         # total points in a hand
PASS          = -1         # sentinel value for "pass" action

assert CARDS_PER_HAND * NUM_PLAYERS + KITTY_SIZE == 50


# ── Deal ──────────────────────────────────────────────────────────────────────

@dataclass
class Deal:
    """Result of a single deal."""
    hands        : list[list[Card]]   # hands[player_index] → list of Card
    kitty        : list[Card]         # 2 face-up cards, part of first trick
    kitty_bonus  : int                # total bonus pts carried by kitty cards
    dealer       : int                # player who dealt (bids last)
    first_bidder : int                # player who bids first (left of dealer)

    def kitty_str(self) -> str:
        parts = []
        for c in self.kitty:
            label = c.short_name
            if c.bonus_points:
                label += f"(+{c.bonus_points})"
            parts.append(label)
        return ", ".join(parts)


def new_deal(dealer: int, rng: Optional[random.Random] = None) -> Deal:
    """
    Shuffle and deal 12 cards to each player; remaining 2 form the kitty.
    Dealer index determines first_bidder = (dealer + 1) % NUM_PLAYERS.
    """
    deck = new_shuffled_deck(rng)
    hands = [deck[i * CARDS_PER_HAND:(i + 1) * CARDS_PER_HAND]
             for i in range(NUM_PLAYERS)]
    kitty = deck[NUM_PLAYERS * CARDS_PER_HAND:]

    assert len(kitty) == KITTY_SIZE
    assert all(len(h) == CARDS_PER_HAND for h in hands)

    kitty_bonus  = sum(c.bonus_points for c in kitty)
    first_bidder = (dealer + 1) % NUM_PLAYERS

    return Deal(
        hands        = hands,
        kitty        = kitty,
        kitty_bonus  = kitty_bonus,
        dealer       = dealer,
        first_bidder = first_bidder,
    )


# ── Bidding state ─────────────────────────────────────────────────────────────

@dataclass
class BiddingState:
    """
    Tracks the live bidding round.

    The bidding sequence visits players in order starting from first_bidder.
    A player's turn produces either a bid (integer >= MIN_BID and > current_bid)
    or a pass (PASS sentinel).

    Bidding ends when:
      a) Three consecutive passes follow the current high bid  → bid_winner set.
      b) All four players pass on the first pass-around       → misdeal.
    """
    first_bidder   : int
    current_bid    : int  = 0           # 0 = no bid yet
    current_bidder : int  = -1          # player who holds the current bid
    bid_history    : list = field(default_factory=list)  # (player, bid|PASS)
    consecutive_passes: int = 0
    bids_placed    : int  = 0           # how many non-pass bids so far
    is_complete    : bool = False
    is_misdeal     : bool = False
    bid_winner     : Optional[int] = None
    winning_bid    : Optional[int] = None

    # Whose turn it is to bid right now
    _turn_index    : int = 0            # index into rotation order

    @property
    def rotation(self) -> list[int]:
        """Players in bid order starting from first_bidder."""
        return [(self.first_bidder + i) % NUM_PLAYERS
                for i in range(NUM_PLAYERS)]

    @property
    def current_player(self) -> int:
        return self.rotation[self._turn_index % NUM_PLAYERS]

    def legal_bids(self) -> list[int]:
        """
        List of legal bid values for the current player.
        Includes PASS (always legal).
        Returns empty list if bidding is already complete.
        """
        if self.is_complete:
            return []
        floor = max(MIN_BID, self.current_bid + 1)
        return [PASS] + list(range(floor, MAX_BID + 1))

    def can_bid(self, amount: int) -> bool:
        return amount == PASS or (
            amount >= MIN_BID
            and amount > self.current_bid
            and amount <= MAX_BID
        )

    def place_bid(self, player: int, amount: int) -> None:
        """
        Record a bid or pass from `player`.
        Raises ValueError on illegal action or wrong player.
        """
        if self.is_complete:
            raise ValueError("Bidding is already complete.")
        if player != self.current_player:
            raise ValueError(
                f"It is Player {self.current_player}'s turn, not Player {player}.")
        if not self.can_bid(amount):
            raise ValueError(
                f"Bid {amount} is illegal. Must be PASS or "
                f">= {max(MIN_BID, self.current_bid + 1)}.")

        self.bid_history.append((player, amount))

        if amount == PASS:
            self.consecutive_passes += 1
        else:
            self.current_bid        = amount
            self.current_bidder     = player
            self.consecutive_passes = 0
            self.bids_placed       += 1

        self._turn_index += 1
        self._check_completion()

    def _check_completion(self) -> None:
        # All four passed without a bid → misdeal
        if (self.bids_placed == 0
                and self.consecutive_passes == NUM_PLAYERS):
            self.is_complete = True
            self.is_misdeal  = True
            return

        # Three consecutive passes after a bid → bidding over
        if self.bids_placed > 0 and self.consecutive_passes == NUM_PLAYERS - 1:
            self.is_complete  = True
            self.bid_winner   = self.current_bidder
            self.winning_bid  = self.current_bid

    def summary(self) -> str:
        lines = [f"Bidding (first bidder: P{self.first_bidder})"]
        for player, amount in self.bid_history:
            label = "PASS" if amount == PASS else str(amount)
            lines.append(f"  P{player}: {label}")
        if self.is_misdeal:
            lines.append("  → MISDEAL (all passed)")
        elif self.is_complete:
            lines.append(
                f"  → P{self.bid_winner} wins bid of {self.winning_bid}")
        else:
            lines.append(f"  → Waiting for P{self.current_player} "
                         f"(current bid: {self.current_bid or 'none'})")
        return "\n".join(lines)


def new_bidding(first_bidder: int) -> BiddingState:
    return BiddingState(first_bidder=first_bidder, _turn_index=0)


# ── Trump declaration (follows bid) ──────────────────────────────────────────

@dataclass
class TrumpDeclaration:
    """The bid winner's trump choice after winning the bid."""
    bid_winner  : int
    trump_suit  : Optional[str]    # None = no-trump declared
    no_trump    : bool             # True if no-trump chosen

    def __str__(self):
        if self.no_trump:
            return f"Player {self.bid_winner} declares NO TRUMP"
        return f"Player {self.bid_winner} declares trump: {self.trump_suit}"


VALID_TRUMP_SUITS = ["Zeros", "Twos", "Fours", "Sixes",
                     "Eights", "Tens", "Twelves"]

def declare_trump(bid_winner: int,
                  trump_suit: Optional[str]) -> TrumpDeclaration:
    """Validate and record the trump declaration."""
    if trump_suit is not None and trump_suit not in VALID_TRUMP_SUITS:
        raise ValueError(f"'{trump_suit}' is not a valid trump suit.")
    return TrumpDeclaration(
        bid_winner = bid_winner,
        trump_suit = trump_suit,
        no_trump   = trump_suit is None,
    )


# ── Observation vector helpers ────────────────────────────────────────────────

def bidding_observation(
    player       : int,
    hand         : list[Card],
    kitty        : list[Card],
    bidding      : BiddingState,
) -> dict:
    """
    Return a dict of everything the current bidder can observe.
    This will later be flattened into the Gym observation vector.

    Visible information during bidding:
      - Own 12-card hand
      - Both kitty cards (face-up)
      - Current high bid
      - Which players have passed / bid (but not their hands)
    """
    pass_flags = [False] * NUM_PLAYERS
    bid_amounts = [0] * NUM_PLAYERS
    for p, amt in bidding.bid_history:
        if amt == PASS:
            pass_flags[p] = True
        else:
            bid_amounts[p] = amt

    return {
        "player"        : player,
        "hand"          : hand,
        "kitty"         : kitty,
        "kitty_bonus"   : sum(c.bonus_points for c in kitty),
        "current_bid"   : bidding.current_bid,
        "current_bidder": bidding.current_bidder,
        "pass_flags"    : pass_flags,
        "bid_amounts"   : bid_amounts,
        "legal_bids"    : bidding.legal_bids(),
    }


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from somerset_deck import hand_str

    rng    = random.Random(42)
    dealer = 0
    deal   = new_deal(dealer, rng)

    print("═" * 55)
    print(f"New deal  (dealer: P{deal.dealer}, "
          f"first bidder: P{deal.first_bidder})")
    print(f"Kitty: {deal.kitty_str()}  "
          f"(+{deal.kitty_bonus} bonus pts)")
    for i, hand in enumerate(deal.hands):
        hand_bonus = sum(c.bonus_points for c in hand)
        print(f"\nPlayer {i} hand ({len(hand)} cards, "
              f"+{hand_bonus} bonus pts):")
        print(hand_str(hand))

    # ── Scenario 1: normal competitive bidding ────────────────────────
    print("\n" + "═" * 55)
    print("Scenario 1: competitive bidding")
    b = new_bidding(deal.first_bidder)
    sequence = [
        (1, 8),
        (2, PASS),
        (3, 14),
        (0, 16),
        (1, PASS),
        (2, PASS),
        (3, PASS),
    ]
    for player, amount in sequence:
        b.place_bid(player, amount)
        if b.is_complete:
            break
    print(b.summary())
    if b.bid_winner is not None:
        td = declare_trump(b.bid_winner, "Eights")
        print(td)

    # ── Scenario 2: misdeal (all pass) ───────────────────────────────
    print("\n" + "═" * 55)
    print("Scenario 2: misdeal — all players pass")
    b2 = new_bidding(deal.first_bidder)
    for player in b2.rotation:
        b2.place_bid(player, PASS)
        if b2.is_complete:
            break
    print(b2.summary())

    # ── Scenario 3: no-trump declaration ─────────────────────────────
    print("\n" + "═" * 55)
    print("Scenario 3: no-trump declared")
    b3 = new_bidding(deal.first_bidder)
    b3.place_bid(1, 10)
    b3.place_bid(2, PASS)
    b3.place_bid(3, PASS)
    b3.place_bid(0, PASS)
    print(b3.summary())
    if b3.bid_winner is not None:
        td3 = declare_trump(b3.bid_winner, None)
        print(td3)

    # ── MIN_BID enforcement ───────────────────────────────────────────
    print("\n" + "═" * 55)
    print(f"Minimum bid enforcement (MIN_BID = {MIN_BID})")
    b4 = new_bidding(0)
    try:
        b4.place_bid(0, 3)   # below minimum
    except ValueError as e:
        print(f"  Correctly rejected bid of 3: {e}")
    b4.place_bid(0, MIN_BID)
    print(f"  Bid of {MIN_BID} accepted. Legal bids now start at {MIN_BID+1}.")
