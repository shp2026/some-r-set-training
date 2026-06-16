"""
somerset_scoring.py
Hand and game scoring logic for Some-R-Set.

Scoring summary:
  - 12 tricks × 1 pt each   = 12 pts
  - Bonus points in deck    = 12 pts  (3+1+1+1+2+2+2)
  - Total per hand          = 25 pts

Teams:
  Team 0: players 0 and 2
  Team 1: players 1 and 3

Bid outcome:
  - Bidding team scores >= bid  → bidding team gets their points; other team gets 0
  - Bidding team scores < bid   → bidding team loses 2 × (bid - scored);
                                   other team gets all their own points

Win condition: first team to reach 66+ points wins the game.
"""

from dataclasses import dataclass, field
from typing import Optional
from somerset_deck import TOTAL_BONUS_POINTS

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_PLAYERS   = 4
NUM_TEAMS     = 2
WINNING_SCORE = 66
MINIMUM_SCORE = -99.  # If a score goes this low, the game ends. 

# Team membership: PLAYER_TEAM[player_index] → team index
PLAYER_TEAM = {0: 0, 1: 1, 2: 0, 3: 1}
TEAM_PLAYERS = {0: [0, 2], 1: [1, 3]}

# Total points available each hand
# 13 tricks (one per card per player in a 4-player deal of 50-card deck
# isn't exactly 13 — let's compute properly)
# 50 cards / 4 players = 12.5, so players get 13/13/12/12 cards.
# Number of tricks = cards in shortest hand = 12.
# But extra cards (players 0 & 1 have 13) mean one player runs out last.
# Actually in trick-taking: tricks played = min hand size ... let's just
# track empirically and assert total == 24 as stated in the rules.
POINTS_PER_TRICK  = 1
TOTAL_HAND_POINTS = 24          # as specified in the rules
PENALTY_MULTIPLIER = 2          # missing bid costs 2× the shortfall


# ── Per-hand result dataclass ─────────────────────────────────────────────────

@dataclass
class TrickResult:
    """Record of a single completed trick."""
    trick_number   : int
    winning_player : int
    winning_team   : int
    cards_played   : list          # list[Card]
    bonus_points   : int


@dataclass
class HandResult:
    """
    Full scoring result for one hand.
    Produced by score_hand(); consumed by apply_hand_result().
    """
    # Inputs
    trick_results   : list[TrickResult]
    bid_value       : int
    bid_winner      : int          # player index who won the bid
    bid_team        : int          # team index of bid winner

    # Computed totals
    tricks_won      : dict = field(default_factory=dict)   # player → count
    team_tricks     : dict = field(default_factory=dict)   # team   → count
    team_raw_points : dict = field(default_factory=dict)   # team   → pts before bid adj
    team_delta      : dict = field(default_factory=dict)   # team   → score change
    bid_made        : bool = False
    shortfall       : int  = 0
    total_points_counted: int = 0

    def summary(self) -> str:
        lines = [
            f"  Bid: {self.bid_value} by Player {self.bid_winner} "
            f"(Team {self.bid_team})",
            f"  Bid made: {self.bid_made}  "
            f"(shortfall: {self.shortfall})",
            f"  Total points counted: {self.total_points_counted}",
            f"  Team raw points: {dict(self.team_raw_points)}",
            f"  Score deltas:    {dict(self.team_delta)}",
        ]
        return "\n".join(lines)


# ── Core scoring function ─────────────────────────────────────────────────────

def score_hand(
    trick_results : list[TrickResult],
    bid_value     : int,
    bid_winner    : int,
) -> HandResult:
    """
    Compute the score delta for each team after a completed hand.

    Parameters
    ----------
    trick_results : list of TrickResult, one per trick played
    bid_value     : the winning bid amount
    bid_winner    : player index who won the bid

    Returns
    -------
    HandResult with all fields populated
    """
    bid_team = PLAYER_TEAM[bid_winner]

    # ── Tally tricks and points per team ─────────────────────────────
    tricks_won      = {p: 0 for p in range(NUM_PLAYERS)}
    team_tricks     = {t: 0 for t in range(NUM_TEAMS)}
    team_raw_points = {t: 0 for t in range(NUM_TEAMS)}

    for tr in trick_results:
        p = tr.winning_player
        t = tr.winning_team
        tricks_won[p]       += 1
        team_tricks[t]      += 1
        team_raw_points[t]  += POINTS_PER_TRICK + tr.bonus_points

    total = sum(team_raw_points.values())

    # Sanity check — warn but don't crash (rules say 24, verify empirically)
    if total != TOTAL_HAND_POINTS:
        import warnings
        warnings.warn(
            f"Expected {TOTAL_HAND_POINTS} total hand points, got {total}. "
            "Check trick bonus tallying or number of tricks played."
        )

    # ── Apply bid outcome ─────────────────────────────────────────────
    bid_team_points = team_raw_points[bid_team]
    other_team      = 1 - bid_team
    other_points    = team_raw_points[other_team]

    bid_made  = bid_team_points >= bid_value
    shortfall = max(0, bid_value - bid_team_points)

    team_delta = {}
    if bid_made:
        # Bidding team earns their points; other team earns nothing this hand
        team_delta[bid_team]  =  bid_team_points
        team_delta[other_team] = 0
    else:
        # Bidding team penalised; other team keeps their raw score
        team_delta[bid_team]   = -(shortfall * PENALTY_MULTIPLIER)
        team_delta[other_team] =  other_points

    result = HandResult(
        trick_results        = trick_results,
        bid_value            = bid_value,
        bid_winner           = bid_winner,
        bid_team             = bid_team,
        tricks_won           = tricks_won,
        team_tricks          = team_tricks,
        team_raw_points      = team_raw_points,
        team_delta           = team_delta,
        bid_made             = bid_made,
        shortfall            = shortfall,
        total_points_counted = total,
    )
    return result


# ── Game state ────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    """
    Tracks cumulative scores across hands and whose turn it is to bid first.

    Attributes
    ----------
    team_scores      : running total for each team (can go negative)
    hand_number      : how many hands have been played
    first_bidder     : player index who bids first this hand
                       (rotates by 1 each hand)
    hand_history     : list of HandResult objects, one per completed hand
    winning_team     : None until the game ends
    """
    team_scores  : dict = field(default_factory=lambda: {0: 0, 1: 0})
    hand_number  : int  = 0
    first_bidder : int  = 0
    hand_history : list = field(default_factory=list)
    winning_team : Optional[int] = None

    def apply_hand_result(self, result: HandResult) -> None:
        """Update game state after a completed hand."""
        for team, delta in result.team_delta.items():
            self.team_scores[team] += delta

        self.hand_history.append(result)
        self.hand_number  += 1
        self.first_bidder  = (self.first_bidder + 1) % NUM_PLAYERS

        # Check win condition
        for team, score in self.team_scores.items():
            if score >= WINNING_SCORE:
                # If both teams hit winning score on the same hand, higher score wins
                if self.winning_team is None:
                    self.winning_team = team
                else:
                    # Already found one — pick the higher scorer
                    other = 1 - team
                    if self.team_scores[team] > self.team_scores[other]:
                        self.winning_team = team
            elif score <= MINIMUM_SCORE:
                # If both teams hit minimum score on the same hand, higher score wins
                if self.winning_team is None:
                    self.winning_team = (team + 1) % 2
                else:
                    # Already found one — pick the higher scorer
                    other = 1 - team
                    if self.team_scores[team] > self.team_scores[other]:
                        self.winning_team = team


    @property
    def game_over(self) -> bool:
        return self.winning_team is not None

    def score_str(self) -> str:
        return (f"Hand {self.hand_number} | "
                f"Team 0 (P0,P2): {self.team_scores[0]:+d}  "
                f"Team 1 (P1,P3): {self.team_scores[1]:+d}  "
                f"Next first bidder: P{self.first_bidder}")


# ── Helper: build TrickResult from raw trick data ─────────────────────────────

def make_trick_result(
    trick_number   : int,
    winning_player : int,
    cards_played   : list,        # list[Card]
) -> TrickResult:
    """Convenience constructor — computes bonus and team automatically."""
    bonus = sum(c.bonus_points for c in cards_played)
    return TrickResult(
        trick_number   = trick_number,
        winning_player = winning_player,
        winning_team   = PLAYER_TEAM[winning_player],
        cards_played   = cards_played,
        bonus_points   = bonus,
    )


# ── Demo / tests ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from somerset_deck import SUIT_VALUE_TO_CARD as SV, INDEX_TO_CARD

    def card(suit, val):  return SV[(suit, val)]
    def ss():             return INDEX_TO_CARD[49]

    game = GameState()
    print(f"Starting game.  Win target: {WINNING_SCORE} pts\n")

    # ── Hand 1: bid made ──────────────────────────────────────────────
    # Team 0 bids 10, scores 18 (bid made)
    # We'll fake 12 tricks with carefully chosen bonus cards
    tricks_h1 = []
    # Team 0 wins tricks 0-7 (8 tricks), Team 1 wins tricks 8-11 (4 tricks)
    # Bonus cards: 6/12, 4/8(+2) and S/S(+3) go to Team 0; 
    #            : 1/2, 2/4, 3/6(+1) and 5/10(+2) to Team 1
    # Team 0 raw = 8 tricks + 2 + 2 + 3 = 15 pts
    # Team 1 raw = 4 tricks + 1 + 1 + 1 +2 =  9 pts   → total 24 ✓

    bonus_to_t0 = [
        [card("Eights", 4)],           # +2 pts  → trick 2
        [ss()],                        # +3 pts  → trick 5
        [card("Twelves", 6)],          # +2 pts  → trick 6
    ]
    bonus_to_t1 = [
        [card("Twos",   1)],           # +1 pt   → trick 8
        [card("Fours",  2)],           # +1 pt   → trick 9
        [card("Tens",   5)],           # +2 pts  → trick 10
        [card("Sixes",  3)],           # +1 pt   → trick 11
    ]

    for i in range(12):
        if i < 8:
            extra = bonus_to_t0.pop(0) if bonus_to_t0 and i in [2,5,6] else []
            winner = 0 if i % 2 == 0 else 2   # alternates between team 0 players
            played = [card("Twelves", i % 6)] + extra
        else:
            extra = bonus_to_t1.pop(0) if bonus_to_t1 else []
            winner = 1 if i % 2 == 0 else 3   # alternates between team 1 players
            played = [card("Tens", i % 5)] + extra
        tricks_h1.append(make_trick_result(i, winner, played))

    result1 = score_hand(tricks_h1, bid_value=10, bid_winner=0)
    game.apply_hand_result(result1)
    print("── Hand 1: Team 0 bids 10 ──────────────────────────────────")
    print(result1.summary())
    print(f"  {game.score_str()}\n")

    # ── Hand 2: bid missed ────────────────────────────────────────────
    # Team 1 bids 16 but scores only 10 → shortfall 6 → penalty -12
    # Team 0 scores 14
    tricks_h2 = []
    bonus_to_t0 = [
        [card("Eights", 4)],           # +2 pts  → trick 2
        [ss()],                        # +3 pts  → trick 5
    ]
    bonus_to_t1 = [
        [card("Twos",   1)],           # +1 pt   → trick 8
        [card("Fours",  2)],           # +1 pt   → trick 9
        [card("Twelves", 6)],          # +2 pts  → trick 6
        [card("Tens",   5)],           # +2 pts  → trick 10
        [card("Sixes",  3)],           # +1 pt   → trick 11
    ]
    for i in range(12):
        if i < 5:   # Team 0 wins first 5
            winner = 0
            extra = bonus_to_t0.pop(0) if bonus_to_t0 else []
            played = [card("Twelves", i % 6)] + extra
        else:        # Team 1 wins last 7
            winner = 1
            extra = bonus_to_t1.pop(0) if bonus_to_t1 else []
            played = [card("Tens", i % 5)] + extra
        tricks_h2.append(make_trick_result(i, winner, played))

    result2 = score_hand(tricks_h2, bid_value=16, bid_winner=1)
    game.apply_hand_result(result2)
    print("── Hand 2: Team 1 bids 16, misses ──────────────────────────")
    print(result2.summary())
    print(f"  {game.score_str()}\n")

    # ── Simulate hands until someone wins ────────────────────────────
    print("── Simulating remaining hands (Team 0 bids 12, makes it) ───!!!!")
    import random
    rng = random.Random(7)
    hand = 3
    while not game.game_over:
        # fake hand: Team 0 always wins 8 tricks + 7 bonus pts
        bonus_to_t0 = [
            [card("Twos",   1)],           # +1 pt
            [card("Fours",  2)],           # +1 pt
            [card("Twelves", 6)],          # +2 pts
            [card("Tens",   5)],           # +2 pts
            [card("Sixes",  3)],           # +1 pt
        ]
        bonus_to_t1 = [
            [card("Eights", 4)],           # +2 pts
            [ss()],                        # +3 pts
        ]
        tr = []
        for i in range(12):
            winner = 0 if i < 8 else 1
            played = [card("Twelves", 0)]
            if winner == 0:
                played += bonus_to_t0.pop(0) if bonus_to_t0 else []
            else:
                played += bonus_to_t1.pop(0) if bonus_to_t1 else []
            tr.append(make_trick_result(i, winner, played))
        res = score_hand(tr, bid_value=10, bid_winner=0)
        game.apply_hand_result(res)
        print(f"  After hand {hand}: {game.score_str()}")
        hand += 1

    print(f"\n🏆  Game over — Team {game.winning_team} wins!")
    print(f"  Final scores: Team 0 = {game.team_scores[0]}, "
          f"Team 1 = {game.team_scores[1]}")
    print(f"  Total hands played: {game.hand_number}")
