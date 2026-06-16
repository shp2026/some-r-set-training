"""
somerset_maenv.py
PettingZoo AEC (Agent-Environment-Cycle) environment for Some-R-Set.

Why AEC instead of Parallel?
  Some-R-Set is a sequential turn-based game — only one player acts at a time.
  AEC is the natural fit: it cycles through agents one at a time, exactly
  mirroring how card games are played.  Parallel environments (where all
  agents act simultaneously) are better suited to RTS or physics sims.

API contract (PettingZoo AEC):
  env.reset()
  while env.agents:                      # non-empty until game over
      agent = env.agent_selection        # whose turn it is
      obs   = env.observe(agent)         # what that agent sees
      mask  = env.action_mask(agent)     # which actions are legal
      action = policy[agent](obs, mask)
      env.step(action)
      reward = env.rewards[agent]        # reward after this step
      done   = env.terminations[agent]

Observation and action spaces are identical to SomeRSetEnv (223-dim obs,
Discrete(50) actions) so a model trained in the single-agent env can be
loaded directly and used here without any architecture changes.
"""

import random
import warnings
import numpy as np
from typing import Optional

import gymnasium
from gymnasium import spaces
from pettingzoo import AECEnv

from somerset_deck   import INDEX_TO_CARD
from somerset_bidding import (
    VALID_TRUMP_SUITS,
    new_deal, new_bidding, declare_trump,
    PASS, MIN_BID, MAX_BID, NUM_PLAYERS,
    CARDS_PER_HAND, VALID_TRUMP_SUITS,
)
from somerset_tricks  import legal_plays, trick_winner
from somerset_scoring import (
    GameState, score_hand, make_trick_result,
    PLAYER_TEAM, WINNING_SCORE,
)
from somerset_env import (
    # Phase constants
    PHASE_BIDDING, PHASE_DECLARING, PHASE_PLAYING, PHASE_SCORING,
    # Action encoding helpers
    bid_to_action, action_to_bid, suit_to_action, action_to_suit,
    BID_ACTION_PASS, DECLARE_NO_TRUMP_ACTION,
    # Observation layout constants
    NUM_CARDS, OBS_TOTAL,
    OBS_HAND_START, OBS_KITTY_START, OBS_SEEN_START, OBS_TRICK_START,
    OBS_TRUMP_START, OBS_HIGH_BID, OBS_PASS_FLAGS,
    OBS_SCORES, OBS_WHOSE_TURN, OBS_PHASE,
)


AGENTS = [f"player_{i}" for i in range(NUM_PLAYERS)]
AGENT_TO_IDX = {a: i for i, a in enumerate(AGENTS)}


class SomeRSetMAEnv(AECEnv):
    """
    PettingZoo AEC environment for Some-R-Set.

    Four agents (player_0 … player_3) form two teams:
      Team 0: player_0, player_2
      Team 1: player_1, player_3

    Rewards are issued at the end of each hand (zero during play):
      - Teammates receive the same delta (team_delta from score_hand).
      - Opponents receive their team's delta.
    The episode ends when one team reaches WINNING_SCORE (66).

    Parameters
    ----------
    render_mode : 'human' for text output, None for silent
    """

    metadata = {
        "render_modes": ["human"],
        "name": "some_r_set_v0",
        "is_parallelizable": False,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        self.possible_agents = AGENTS[:]
        self.agents          = []

        # Spaces — identical layout to single-agent env
        obs_low  = np.zeros(OBS_TOTAL, dtype=np.float32)
        obs_high = np.ones(OBS_TOTAL,  dtype=np.float32)
        obs_low[OBS_SCORES: OBS_SCORES + 2] = -1.0

        self._obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self._act_space = spaces.Discrete(NUM_CARDS)

        # Runtime state
        self.game           = None
        self.deal           = None
        self.bid            = None
        self.trump          = None
        self.phase          = PHASE_BIDDING
        self.current_player = 0
        self.trick_number   = 0
        self.current_trick  = []
        self.trick_results  = []
        self.cards_seen     = set()
        self.dealer         = 0

    # ── PettingZoo required properties ───────────────────────────────────────

    @property
    def observation_spaces(self):
        return {a: self._obs_space for a in self.possible_agents}

    @property
    def action_spaces(self):
        return {a: self._act_space for a in self.possible_agents}

    def observation_space(self, agent: str):
        return self._obs_space

    def action_space(self, agent: str):
        return self._act_space

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = self.possible_agents[:]

        self.rewards        = {a: 0.0 for a in self.agents}
        self.terminations   = {a: False for a in self.agents}
        self.truncations    = {a: False for a in self.agents}
        self.infos          = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}

        if self.game is None:
            self.game   = GameState()
            self.dealer = 0
        else:
            self.dealer = (self.dealer + 1) % NUM_PLAYERS

        self._start_new_hand()

        # Set agent_selection to who acts first
        self.agent_selection = AGENTS[self.current_player]

        if self.render_mode == "human":
            self.render()

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        agent = self.agent_selection
        player = AGENT_TO_IDX[agent]

        # Clear last rewards
        self.rewards = {a: 0.0 for a in self.agents}

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Validate and apply
        legal = self._legal_actions(player)
        if action not in legal:
            # Illegal move — small penalty, don't advance
            self.rewards[agent] = -1.0
            self._cumulative_rewards[agent] += -1.0
            return

        self._apply_action(player, action)

        # If scoring phase, resolve and issue rewards
        if self.phase == PHASE_SCORING:
            self._do_scoring()

        # Advance agent_selection to next actor
        self.agent_selection = AGENTS[self.current_player]

        # Accumulate rewards
        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]

        if self.render_mode == "human":
            self.render()

    # ── observe ───────────────────────────────────────────────────────────────

    def observe(self, agent: str) -> np.ndarray:
        player = AGENT_TO_IDX[agent]
        return self._get_observation(player)

    def action_mask(self, agent: str) -> np.ndarray:
        player = AGENT_TO_IDX[agent]
        legal  = self._legal_actions(player)
        mask   = np.zeros(NUM_CARDS, dtype=bool)
        for a in legal:
            mask[a] = True
        return mask

    # ── render ────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode != "human":
            return
        from somerset_deck import hand_str
        phase_names = ["BIDDING","DECLARING","PLAYING","SCORING"]
        g = self.game
        print(f"\n{'─'*55}")
        print(f"Hand {g.hand_number+1} | {phase_names[self.phase]} | "
              f"Trick {self.trick_number+1}/12 | "
              f"Acting: {self.agent_selection}")
        print(f"Scores → Team0: {g.team_scores[0]:+d}  "
              f"Team1: {g.team_scores[1]:+d}")
        if self.trump:
            ts = self.trump.trump_suit or "NO TRUMP"
            print(f"Trump: {ts}  Bid: {self.bid.winning_bid} "
                  f"by {AGENTS[self.bid.bid_winner]}")
        if self.deal:
            print(f"Kitty: {self.deal.kitty_str()}")

    # ── Internal: hand setup ──────────────────────────────────────────────────

    def _start_new_hand(self):
        rng = random.Random()
        self.deal           = new_deal(self.dealer, rng)
        self.bid            = new_bidding(self.deal.first_bidder)
        self.trump          = None
        self.phase          = PHASE_BIDDING
        self.current_player = self.deal.first_bidder
        self.trick_number   = 0
        self.current_trick  = []
        self.trick_results  = []
        self.cards_seen     = set()

    # ── Internal: action dispatch ─────────────────────────────────────────────

    def _apply_action(self, player: int, action: int):
        if self.phase == PHASE_BIDDING:
            self._apply_bid(player, action)
        elif self.phase == PHASE_DECLARING:
            self._apply_declare(player, action)
        elif self.phase == PHASE_PLAYING:
            self._apply_play(player, action)

    def _apply_bid(self, player: int, action: int):
        amount = action_to_bid(action)
        self.bid.place_bid(player, amount)

        if self.bid.is_complete:
            if self.bid.is_misdeal:
                self._start_new_hand()
            else:
                self.phase          = PHASE_DECLARING
                self.current_player = self.bid.bid_winner
        else:
            self.current_player = self.bid.current_player

    def _apply_declare(self, player: int, action: int):
        trump_suit      = action_to_suit(action)
        self.trump      = declare_trump(player, trump_suit)
        self.phase      = PHASE_PLAYING
        self.current_trick = [(c, -1) for c in self.deal.kitty]
        self.current_player = self.bid.bid_winner

    def _apply_play(self, player: int, action: int):
        card = INDEX_TO_CARD[action]
        self.deal.hands[player].remove(card)
        self.current_trick.append((card, player))

        players_in_trick = sum(1 for _, p in self.current_trick if p >= 0)
        if players_in_trick == NUM_PLAYERS:
            self._resolve_trick()
        else:
            self.current_player = (player + 1) % NUM_PLAYERS

    def _resolve_trick(self):
        trump_suit = self.trump.trump_suit if self.trump else None
        winner_card, winner_player = trick_winner(self.current_trick, trump_suit)

        if winner_player < 0:
            winner_player = self.bid.bid_winner

        cards = [c for c, _ in self.current_trick]
        tr    = make_trick_result(self.trick_number, winner_player, cards)
        self.trick_results.append(tr)

        for c, p in self.current_trick:
            if p >= 0:
                self.cards_seen.add(c.index)

        self.trick_number  += 1
        self.current_trick  = []

        if self.trick_number == 12:
            self.phase = PHASE_SCORING
            # current_player doesn't matter; scoring will reset it
        else:
            self.current_player = winner_player

    def _do_scoring(self):
        result = score_hand(
            self.trick_results,
            bid_value  = self.bid.winning_bid,
            bid_winner = self.bid.bid_winner,
        )
        self.game.apply_hand_result(result)

        # Issue rewards to all agents
        for agent in self.agents:
            p    = AGENT_TO_IDX[agent]
            team = PLAYER_TEAM[p]
            self.rewards[agent] = float(result.team_delta[team])

        if self.game.game_over:
            for a in self.agents:
                self.terminations[a] = True
            # Remove finished agents from active list
            self.agents = []
        else:
            # Start next hand
            self.dealer = (self.dealer + 1) % NUM_PLAYERS
            self._start_new_hand()
            self.phase = PHASE_BIDDING

    # ── Internal: legal actions ───────────────────────────────────────────────

    def _legal_actions(self, player: int) -> list[int]:
        if self.phase == PHASE_BIDDING:
            return [bid_to_action(b) for b in self.bid.legal_bids()]

        if self.phase == PHASE_DECLARING:
            return list(range(len(VALID_TRUMP_SUITS) + 1))

        if self.phase == PHASE_PLAYING:
            trump_suit = self.trump.trump_suit if self.trump else None
            hand       = self.deal.hands[player]
            led_card   = next(
                (c for c, p in self.current_trick if p >= 0), None)
            plays = legal_plays(
                hand           = hand,
                led_card       = led_card,
                trump_suit     = trump_suit,
                is_first_trick = self.trick_number == 0,
                is_leading     = led_card is None,
            )
            return [c.index for c in plays]

        return []

    # ── Internal: observation ─────────────────────────────────────────────────

    def _get_observation(self, player: int) -> np.ndarray:
        obs = np.zeros(OBS_TOTAL, dtype=np.float32)
        g   = self.game

        # Hand
        if self.deal:
            for c in self.deal.hands[player]:
                obs[OBS_HAND_START + c.index] = 1.0
            # Kitty
            for c in self.deal.kitty:
                obs[OBS_KITTY_START + c.index] = 1.0

        # Seen cards
        for idx in self.cards_seen:
            obs[OBS_SEEN_START + idx] = 1.0

        # Current trick
        for c, _ in self.current_trick:
            obs[OBS_TRICK_START + c.index] = 1.0

        # Trump
        if self.trump is not None:
            if self.trump.no_trump:
                obs[OBS_TRUMP_START + 7] = 1.0
            else:
                idx = VALID_TRUMP_SUITS.index(self.trump.trump_suit)
                obs[OBS_TRUMP_START + idx] = 1.0

        # High bid
        if self.bid:
            obs[OBS_HIGH_BID] = self.bid.current_bid / MAX_BID
            for p, amt in self.bid.bid_history:
                if amt == PASS:
                    obs[OBS_PASS_FLAGS + p] = 1.0

        # Scores (clipped to [-1, 1])
        for t in range(2):
            raw = g.team_scores[t] / WINNING_SCORE
            obs[OBS_SCORES + t] = np.clip(raw, -1.0, 1.0)

        # Whose turn
        obs[OBS_WHOSE_TURN + self.current_player] = 1.0

        # Phase
        obs[OBS_PHASE + min(self.phase, 3)] = 1.0

        return obs
