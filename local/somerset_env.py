"""
somerset_env.py
Full Gymnasium environment for Some-R-Set.

Phase flow each episode:
  DEALING      → new_deal()         : shuffle, deal 12 cards each + 2 kitty
  BIDDING      → BiddingState       : players bid or pass in rotation
  DECLARING    → TrumpDeclaration   : bid winner names trump (or no-trump)
  PLAYING      → trick loop x 12    : players play cards trick by trick
  SCORING      → score_hand()       : tally points, apply bid outcome
  (repeat from DEALING until a team reaches 66 pts)

Single-agent design:
  The environment manages all four players, but exposes one agent seat
  (default: player 0) as the "learning agent". The other three players
  are controlled by a pluggable OpponentPolicy (random by default).
  All observations are from the perspective of the learning agent.

Observation vector layout  (float32, all values in [0, 1]):
  Segment                        Size   Notes
  ─────────────────────────────────────────────────────────────
  My hand (card present)           50   one-hot per card index
  Kitty cards                      50   one-hot; face-up all game
  Cards seen (played in prior tricks) 50
  Cards in current trick           50   includes kitty in trick 1
  Trump suit (one-hot)              8   7 suits + no-trump flag
  Current high bid (normalised)     1   bid / MAX_BID
  Player pass flags                 4   1 if player has passed
  Scores team 0 & 1 (normalised)   2   score / WINNING_SCORE
  Whose turn (one-hot)              4
  Phase (one-hot)                   4   BIDDING/DECLARING/PLAYING/SCORING
  ─────────────────────────────────────────────────────────────
  Total                           223

Action space:
  During BIDDING   : Discrete(20)  → 0=PASS, 1=bid6, 2=bid7, … 19=bid24
  During DECLARING : Discrete(8)   → 0..6=suit index, 7=no-trump
  During PLAYING   : Discrete(50)  → card index 0-49

  We use a single Discrete(50) space throughout (largest), and encode
  bidding/declaring actions into that same range.  Legal masks are
  always provided in info["legal_actions"].
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Callable

from somerset_deck import (
    Card, MASTER_DECK, INDEX_TO_CARD, SUIT_DEFS,
    new_shuffled_deck, TOTAL_BONUS_POINTS,
)
from somerset_bidding import (
    new_deal, new_bidding, declare_trump,
    Deal, BiddingState, TrumpDeclaration,
    PASS, MIN_BID, MAX_BID, NUM_PLAYERS,
    CARDS_PER_HAND, KITTY_SIZE, VALID_TRUMP_SUITS,
)
from somerset_tricks import (
    legal_plays, trick_winner, trick_summary,
    CARD_0_0, CARD_SS,
)
from somerset_scoring import (
    GameState, HandResult, TrickResult,
    score_hand, make_trick_result,
    PLAYER_TEAM, TEAM_PLAYERS, WINNING_SCORE,
    POINTS_PER_TRICK, TOTAL_HAND_POINTS,
)

# ── Phase constants ────────────────────────────────────────────────────────────

PHASE_BIDDING   = 0
PHASE_DECLARING = 1
PHASE_PLAYING   = 2
PHASE_SCORING   = 3   # transient; env auto-advances after scoring

# ── Action encoding ────────────────────────────────────────────────────────────
# All phases use a Discrete(50) action space.
# Bidding actions are encoded as:  0=PASS, 1=bid MIN_BID, 2=bid MIN_BID+1, …
# Declaring actions:               0..6 = suit index in VALID_TRUMP_SUITS, 7=no-trump
# Playing actions:                 card index 0-49  (direct)

BID_ACTION_PASS   = 0
BID_ACTION_OFFSET = MIN_BID - 1   # action k → bid value k + BID_ACTION_OFFSET

def bid_to_action(bid: int) -> int:
    if bid == PASS: return BID_ACTION_PASS
    return bid - BID_ACTION_OFFSET        # bid 6 → action 1

def action_to_bid(action: int) -> int:
    if action == BID_ACTION_PASS: return PASS
    return action + BID_ACTION_OFFSET     # action 1 → bid 6

DECLARE_NO_TRUMP_ACTION = len(VALID_TRUMP_SUITS)   # = 7

def suit_to_action(suit: Optional[str]) -> int:
    if suit is None: return DECLARE_NO_TRUMP_ACTION
    return VALID_TRUMP_SUITS.index(suit)

def action_to_suit(action: int) -> Optional[str]:
    if action == DECLARE_NO_TRUMP_ACTION: return None
    return VALID_TRUMP_SUITS[action]

# ── Observation layout ─────────────────────────────────────────────────────────

NUM_CARDS       = 50
NUM_SUITS       = len(VALID_TRUMP_SUITS) + 1   # 7 + no-trump flag = 8

OBS_HAND_START       = 0
OBS_KITTY_START      = 50
OBS_SEEN_START       = 100
OBS_TRICK_START      = 150
OBS_TRUMP_START      = 200
OBS_HIGH_BID         = 208
OBS_PASS_FLAGS       = 209
OBS_SCORES           = 213
OBS_WHOSE_TURN       = 215
OBS_PHASE            = 219
OBS_TOTAL            = 223


# ── Default opponent policy: random legal action ───────────────────────────────

def random_opponent_policy(legal_actions: list[int], **kwargs) -> int:
    return random.choice(legal_actions)


# ── Main environment class ─────────────────────────────────────────────────────

class SomeRSetEnv(gym.Env):
    """
    Gymnasium environment for Some-R-Set (single learning agent vs 3 opponents).

    Parameters
    ----------
    agent_player      : which seat (0-3) is the learning agent
    opponent_policy   : callable(legal_actions, obs, phase) → action int
                        defaults to random_opponent_policy
    render_mode       : 'human' for text output, None for silent
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        agent_player    : int = 0,
        opponent_policy : Callable = random_opponent_policy,
        render_mode     : Optional[str] = None,
    ):
        super().__init__()
        self.agent_player    = agent_player
        self.opponent_policy = opponent_policy
        self.render_mode     = render_mode

        # Single Discrete(50) covers all phases
        self.action_space = spaces.Discrete(NUM_CARDS)

        # Most dims are 0/1 flags, but score dims (OBS_SCORES) can be in [-1,1]
        obs_low  = np.zeros(OBS_TOTAL, dtype=np.float32)
        obs_high = np.ones(OBS_TOTAL,  dtype=np.float32)
        obs_low[OBS_SCORES : OBS_SCORES + 2]  = -1.0
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32)

        # Runtime state (populated in reset)
        self.game : Optional[GameState]       = None
        self.deal : Optional[Deal]            = None
        self.bid  : Optional[BiddingState]    = None
        self.trump: Optional[TrumpDeclaration]= None

        self.phase          : int  = PHASE_BIDDING
        self.current_player : int  = 0
        self.trick_number   : int  = 0
        self.current_trick  : list = []   # list of (Card, player_index)
        self.trick_results  : list = []   # TrickResult per completed trick
        self.cards_seen     : set  = set()# card indices played in prior tricks
        self.dealer         : int  = 0

    # ═══════════════════════════════════════════════════════════════════
    # reset()
    # ═══════════════════════════════════════════════════════════════════

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        py_rng = random.Random(seed)

        if self.game is None:
            self.game   = GameState()
            self.dealer = 0
        else:
            # Dealer rotates each hand (first_bidder rotation already in
            # GameState, but dealer is one seat earlier)
            self.dealer = (self.dealer + 1) % NUM_PLAYERS

        self._start_new_hand(py_rng)

        # Let opponents act until it's the agent's turn
        obs, info = self._advance_to_agent()
        return obs, info

    # ═══════════════════════════════════════════════════════════════════
    # step()
    # ═══════════════════════════════════════════════════════════════════

    def step(self, action: int):
        assert not self.game.game_over, "Game is over; call reset()."

        legal = self._legal_actions()
        if action not in legal:
            # Penalise illegal action and return current state unchanged
            obs  = self._get_observation()
            info = {"legal_actions": legal, "illegal_action": True}
            return obs, -1.0, False, False, info

        self._apply_action(self.agent_player, action)

        # Let opponents respond until it's the agent's turn again
        obs, info = self._advance_to_agent()

        reward     = info.pop("reward", 0.0)
        terminated = self.game.game_over
        truncated  = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ═══════════════════════════════════════════════════════════════════
    # render()
    # ═══════════════════════════════════════════════════════════════════

    def render(self):
        if self.render_mode != "human":
            return
        from somerset_deck import hand_str
        g = self.game
        print(f"\n{'─'*55}")
        phase_names = ["BIDDING","DECLARING","PLAYING","SCORING"]
        print(f"Hand {g.hand_number+1} | Phase: {phase_names[self.phase]} | "
              f"Trick: {self.trick_number+1}/12")
        print(f"Scores → Team 0: {g.team_scores[0]:+d}  "
              f"Team 1: {g.team_scores[1]:+d}")
        if self.trump:
            ts = self.trump.trump_suit or "NO TRUMP"
            print(f"Trump: {ts}  |  Bid: {self.bid.winning_bid} "
                  f"by P{self.bid.bid_winner}")
        print(f"Kitty: {self.deal.kitty_str()}")
        print(f"Agent (P{self.agent_player}) hand:")
        print(hand_str(self.deal.hands[self.agent_player]))
        if self.current_trick:
            played = ", ".join(
                f"P{p}:{c.short_name}" for c, p in self.current_trick
            )
            print(f"Current trick: {played}")

    # ═══════════════════════════════════════════════════════════════════
    # Internal: hand setup
    # ═══════════════════════════════════════════════════════════════════

    def _start_new_hand(self, rng: random.Random):
        self.deal         = new_deal(self.dealer, rng)
        self.bid          = new_bidding(self.deal.first_bidder)
        self.trump        = None
        self.phase        = PHASE_BIDDING
        self.current_player = self.deal.first_bidder
        self.trick_number = 0
        self.current_trick = []
        self.trick_results = []
        self.cards_seen   = set()

    # ═══════════════════════════════════════════════════════════════════
    # Internal: advance through opponent turns until agent's turn
    # ═══════════════════════════════════════════════════════════════════

    def _advance_to_agent(self) -> tuple[np.ndarray, dict]:
        """
        Run opponent actions until the learning agent must act,
        or until the hand/game ends.
        Returns (obs, info) from the agent's perspective.
        """
        reward = 0.0

        while True:
            if self.game.game_over:
                break

            if self.phase == PHASE_SCORING:
                r = self._do_scoring()
                reward += r
                if self.game.game_over:
                    break
                # Start next hand
                self._start_new_hand(random.Random())
                continue

            if self.current_player == self.agent_player:
                break

            # Opponent acts
            legal   = self._legal_actions()
            obs     = self._get_observation()
            action  = self.opponent_policy(
                legal, obs=obs, phase=self.phase)
            self._apply_action(self.current_player, action)

        obs  = self._get_observation()
        info = {
            "legal_actions" : self._legal_actions(),
            "phase"         : self.phase,
            "trick_number"  : self.trick_number,
            "team_scores"   : dict(self.game.team_scores),
            "reward"        : reward,
        }
        return obs, info

    # ═══════════════════════════════════════════════════════════════════
    # Internal: apply one action for one player
    # ═══════════════════════════════════════════════════════════════════

    def _apply_action(self, player: int, action: int):
        if self.phase == PHASE_BIDDING:
            self._apply_bid(player, action)
        elif self.phase == PHASE_DECLARING:
            self._apply_declare(player, action)
        elif self.phase == PHASE_PLAYING:
            self._apply_play(player, action)

    def _apply_bid(self, player: int, action: int):
        bid_amount = action_to_bid(action)
        self.bid.place_bid(player, bid_amount)

        if self.bid.is_complete:
            if self.bid.is_misdeal:
                # Redeal same hand; dealer doesn't rotate on misdeal
                self._start_new_hand(random.Random())
            else:
                self.phase          = PHASE_DECLARING
                self.current_player = self.bid.bid_winner
        else:
            self.current_player = self.bid.current_player

    def _apply_declare(self, player: int, action: int):
        trump_suit  = action_to_suit(action)
        self.trump  = declare_trump(player, trump_suit)
        self.phase  = PHASE_PLAYING

        # First trick starts with the two kitty cards already in it,
        # then the bid winner leads the first actual card.
        self.current_trick = [
            (c, -1) for c in self.deal.kitty   # -1 = kitty (no player)
        ]
        self.current_player = self.bid.bid_winner

    def _apply_play(self, player: int, action: int):
        card = INDEX_TO_CARD[action]
        self.deal.hands[player].remove(card)
        self.current_trick.append((card, player))

        # A trick is complete when all 4 players have contributed.
        # Trick 1 already has 2 kitty cards, so only 4 player cards needed;
        # subsequent tricks need exactly 4.
        players_in_trick = sum(1 for _, p in self.current_trick if p >= 0)

        if players_in_trick == NUM_PLAYERS:
            self._resolve_trick()
        else:
            # Advance to next player
            self.current_player = (player + 1) % NUM_PLAYERS

    def _resolve_trick(self):
        trump_suit = self.trump.trump_suit if self.trump else None

        # Compute winner (kitty cards participate fully)
        winner_card, winner_player = trick_winner(
            self.current_trick, trump_suit)

        # winner_player == -1 means a kitty card won; award to bid winner
        if winner_player < 0:
            winner_player = self.bid.bid_winner

        cards_in_trick = [c for c, _ in self.current_trick]
        tr = make_trick_result(self.trick_number, winner_player, cards_in_trick)
        self.trick_results.append(tr)

        # Mark all non-kitty cards as seen
        for c, p in self.current_trick:
            if p >= 0:
                self.cards_seen.add(c.index)

        self.trick_number  += 1
        self.current_trick  = []

        if self.trick_number == 12:
            self.phase = PHASE_SCORING
        else:
            self.current_player = winner_player

    def _do_scoring(self) -> float:
        """Score the hand, update game state, return reward for agent."""
        result = score_hand(
            self.trick_results,
            bid_value  = self.bid.winning_bid,
            bid_winner = self.bid.bid_winner,
        )
        self.game.apply_hand_result(result)

        agent_team  = PLAYER_TEAM[self.agent_player]
        reward      = float(result.team_delta[agent_team])
        self.phase  = PHASE_BIDDING   # will be overridden by next hand start
        return reward

    # ═══════════════════════════════════════════════════════════════════
    # Internal: legal actions
    # ═══════════════════════════════════════════════════════════════════

    def _legal_actions(self) -> list[int]:
        if self.phase == PHASE_BIDDING:
            return [bid_to_action(b) for b in self.bid.legal_bids()]

        if self.phase == PHASE_DECLARING:
            # Bid winner may name any suit or no-trump
            return list(range(len(VALID_TRUMP_SUITS) + 1))   # 0-7

        if self.phase == PHASE_PLAYING:
            trump_suit   = self.trump.trump_suit if self.trump else None
            hand         = self.deal.hands[self.current_player]
            led_card     = self._led_card()
            is_first     = self.trick_number == 0
            is_leading   = led_card is None

            plays = legal_plays(
                hand          = hand,
                led_card      = led_card,
                trump_suit    = trump_suit,
                is_first_trick= is_first,
                is_leading    = is_leading,
            )
            return [c.index for c in plays]

        return []

    def _led_card(self) -> Optional[Card]:
        """
        The card that sets the led suit for following purposes.
        In trick 1, kitty cards are pre-loaded but don't set the led suit —
        the first card played by a human/agent sets it.
        """
        for card, player in self.current_trick:
            if player >= 0:
                return card
        return None

    # ═══════════════════════════════════════════════════════════════════
    # Internal: observation
    # ═══════════════════════════════════════════════════════════════════

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(OBS_TOTAL, dtype=np.float32)
        ap  = self.agent_player
        g   = self.game

        # My hand
        for c in self.deal.hands[ap]:
            obs[OBS_HAND_START + c.index] = 1.0

        # Kitty (always visible)
        for c in self.deal.kitty:
            obs[OBS_KITTY_START + c.index] = 1.0

        # Cards seen in prior tricks
        for idx in self.cards_seen:
            obs[OBS_SEEN_START + idx] = 1.0

        # Cards in current trick (including kitty in trick 1)
        for c, _ in self.current_trick:
            obs[OBS_TRICK_START + c.index] = 1.0

        # Trump suit one-hot (index 7 = no-trump)
        if self.trump is not None:
            if self.trump.no_trump:
                obs[OBS_TRUMP_START + 7] = 1.0
            else:
                suit_idx = VALID_TRUMP_SUITS.index(self.trump.trump_suit)
                obs[OBS_TRUMP_START + suit_idx] = 1.0

        # Current high bid (normalised)
        obs[OBS_HIGH_BID] = self.bid.current_bid / MAX_BID

        # Pass flags
        for p, amt in self.bid.bid_history:
            if amt == PASS:
                obs[OBS_PASS_FLAGS + p] = 1.0

        # Team scores (normalised, clipped to [-1, 1])
        for t in range(2):
            raw = g.team_scores[t] / WINNING_SCORE
            obs[OBS_SCORES + t] = np.clip(raw, -1.0, 1.0)

        # Whose turn (one-hot)
        obs[OBS_WHOSE_TURN + self.current_player] = 1.0

        # Phase (one-hot)
        obs[OBS_PHASE + self.phase] = 1.0

        return obs


# ═══════════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")   # suppress bonus-point warnings in demo

    print("SomeRSet Gymnasium environment — smoke test")
    print("=" * 55)

    env = SomeRSetEnv(agent_player=0, render_mode="human")
    obs, info = env.reset(seed=42)

    print(f"\nObs shape  : {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Legal actions at start: {info['legal_actions']}")

    # Run one full game with the agent also playing randomly
    total_hands = 0
    total_steps = 0
    episode_reward = 0.0

    obs, info = env.reset(seed=42)

    while not env.game.game_over:
        legal   = info["legal_actions"]
        action  = random.choice(legal)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        total_steps    += 1
        if terminated:
            break

    total_hands = env.game.hand_number
    print(f"\n{'='*55}")
    print(f"Game complete!")
    print(f"  Hands played  : {total_hands}")
    print(f"  Steps taken   : {total_steps}")
    print(f"  Winning team  : Team {env.game.winning_team}")
    print(f"  Final scores  : Team 0 = {env.game.team_scores[0]:+d}, "
          f"Team 1 = {env.game.team_scores[1]:+d}")
    print(f"  Agent reward  : {episode_reward:+.1f}")

    # Verify observation space compliance
    obs, info = env.reset(seed=99)
    assert env.observation_space.contains(obs), "Obs out of bounds!"
    print(f"\nObservation space check: PASSED")
    print(f"Obs vector sample (first 30 dims): {obs[:30]}")
