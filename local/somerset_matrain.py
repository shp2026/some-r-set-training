"""
somerset_matrain.py
Multi-agent self-play training for Some-R-Set using MaskablePPO.

Strategy: Parameter Sharing with Team Roles
  All four players share ONE policy network, but each receives its own
  observation (from its seat's perspective).  This works well for
  symmetric team games:
    - The policy learns to play from any seat
    - Teammates naturally coordinate because they share weights
    - Opponents are also playing the same policy, so training is self-play

  We wrap the AEC env in a "turn-based single-agent adapter" that presents
  one agent's turn at a time to MaskablePPO, cycling through all four players.
  This is the standard approach for AEC envs with SB3.

Phase 1 — Warm start (optional):
  Load the single-agent model trained in somerset_train.py and use its weights
  to initialise all four agents.  This lets multi-agent training start from a
  competent baseline rather than random play.

Phase 2 — Self-play training:
  All four agents improve together.  Because teammates share weights, improving
  one player's bidding also improves their partner's bidding for free.

Usage:
  # Train from scratch:
  python somerset_matrain.py

  # Warm-start from single-agent model:
  python somerset_matrain.py --warmstart ./models/somerset/best_model

  # Continue training an existing MA model:
  python somerset_matrain.py --resume ./models/ma/somerset_ma_final

  # Watch a trained model play:
  python somerset_matrain.py --play ./models/ma/somerset_ma_final
"""

import argparse
import os
import warnings
import numpy as np
import random
from typing import Optional
from collections import deque

import gymnasium
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList
)

from somerset_maenv import SomeRSetMAEnv, AGENTS, AGENT_TO_IDX
from somerset_scoring import PLAYER_TEAM, WINNING_SCORE
from somerset_env import PHASE_BIDDING, PHASE_DECLARING, PHASE_PLAYING

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# Turn-based single-agent adapter
# ═══════════════════════════════════════════════════════════════════════════════

class TurnBasedAdapter(gymnasium.Env):
    """
    Wraps SomeRSetMAEnv so that MaskablePPO sees a single-agent Gym interface.

    The key insight: in a turn-based game, each "step" from the RL algorithm's
    perspective covers exactly ONE player's action.  The adapter presents the
    current player's observation and collects their action.  All four players
    share the same weights (parameter sharing), so the adapter presents each
    seat's perspective in turn without MaskablePPO needing to know there are
    multiple agents.

    Episode boundary: the adapter signals done=True at game end (when one team
    reaches 66).  The reward returned at each step is 0 during the hand, then
    the team's score delta at the end of the hand, matching the MA env.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self._ma_env = SomeRSetMAEnv()
        self._seed   = seed

        # Mirror spaces from MA env
        self.observation_space = self._ma_env.observation_space("player_0")
        self.action_space      = self._ma_env.action_space("player_0")

        # Rolling stats for the callback
        self._ep_rewards  = deque(maxlen=100)
        self._ep_lengths  = deque(maxlen=100)
        self._ep_wins     = deque(maxlen=100)
        self._hand_counts = deque(maxlen=100)

        self._ep_reward = 0.0
        self._ep_steps  = 0

    def reset(self, *, seed=None, options=None):
        s = seed if seed is not None else self._seed
        self._ma_env.reset(seed=s)
        self._ep_reward = 0.0
        self._ep_steps  = 0

        obs  = self._current_obs()
        info = {"action_masks": self._current_mask()}
        return obs, info

    def step(self, action: int):
        self._ma_env.step(action)

        reward     = self._current_reward()
        terminated = len(self._ma_env.agents) == 0
        truncated  = False

        self._ep_reward += reward
        self._ep_steps  += 1

        if terminated:
            self._ep_rewards.append(self._ep_reward)
            self._ep_lengths.append(self._ep_steps)
            win = 1 if self._ma_env.game.winning_team == 0 else 0
            self._ep_wins.append(win)
            self._hand_counts.append(self._ma_env.game.hand_number)

        obs  = self._current_obs()
        info = {"action_masks": self._current_mask()}

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO before sampling."""
        return self._current_mask()

    def _current_obs(self) -> np.ndarray:
        if not self._ma_env.agents:
            # Game over — return zeroed obs
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        agent = self._ma_env.agent_selection
        return self._ma_env.observe(agent)

    def _current_mask(self) -> np.ndarray:
        if not self._ma_env.agents:
            mask    = np.zeros(self.action_space.n, dtype=bool)
            mask[0] = True   # dummy legal action
            return mask
        agent = self._ma_env.agent_selection
        return self._ma_env.action_mask(agent)

    def _current_reward(self) -> float:
        """
        Reward for the agent who just acted.
        MA env stores per-agent rewards; we read the one who just acted.
        agent_selection has already advanced, so we read the previous agent.
        """
        if not self._ma_env.agents:
            # All agents terminated — read cumulative rewards
            # Return team 0's per-hand delta as proxy
            return 0.0
        # Rewards are set during scoring; 0 otherwise
        prev = self._ma_env.agent_selection
        return float(self._ma_env.rewards.get(prev, 0.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Environment factory for vectorised training
# ═══════════════════════════════════════════════════════════════════════════════

def make_ma_env(rank: int, seed: int = 0):
    def _init():
        env = TurnBasedAdapter(seed=seed + rank)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ═══════════════════════════════════════════════════════════════════════════════
# Training callback with MA-specific metrics
# ═══════════════════════════════════════════════════════════════════════════════

class MATrainingCallback(BaseCallback):
    """
    Logs multi-agent specific metrics every N rollouts:
      - Mean episode reward (team 0 perspective)
      - Win rate for team 0
      - Mean hands per game
    """
    def __init__(self, log_freq: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq   = log_freq
        self.n_rollouts = 0

    def _on_rollout_end(self) -> bool:
        self.n_rollouts += 1
        if self.n_rollouts % self.log_freq != 0:
            return True

        ep_infos = self.model.ep_info_buffer
        if not ep_infos:
            return True

        rewards = [ep["r"] for ep in ep_infos]
        lengths = [ep["l"] for ep in ep_infos]

        self.logger.record("ma/mean_ep_reward", np.mean(rewards))
        self.logger.record("ma/mean_ep_length", np.mean(lengths))

        if self.verbose >= 1:
            print(f"  Rollout {self.n_rollouts:4d} | "
                  f"reward={np.mean(rewards):+.1f} | "
                  f"ep_len={np.mean(lengths):.0f}")
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation: run N full games, track win rate and score margin
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_ma(model: MaskablePPO, n_games: int = 30) -> dict:
    """
    Evaluate the model by running complete games.
    Returns win_rate, mean_score_margin, mean_hands.
    """
    wins, margins, hand_counts = [], [], []

    for g in range(n_games):
        env = SomeRSetMAEnv()
        env.reset(seed=2000 + g)

        while env.agents:
            agent = env.agent_selection
            obs   = env.observe(agent)
            mask  = env.action_mask(agent)
            action, _ = model.predict(
                obs[np.newaxis], action_masks=mask[np.newaxis],
                deterministic=True)
            env.step(int(action[0]))

        winner = env.game.winning_team
        scores = env.game.team_scores
        wins.append(1 if winner == 0 else 0)
        margins.append(scores[0] - scores[1])
        hand_counts.append(env.game.hand_number)

    return {
        "win_rate"     : np.mean(wins),
        "mean_margin"  : np.mean(margins),
        "mean_hands"   : np.mean(hand_counts),
        "n_games"      : n_games,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main training routine
# ═══════════════════════════════════════════════════════════════════════════════

def train_ma(
    total_steps    : int   = 1_000_000,
    n_envs         : int   = 4,
    seed           : int   = 42,
    warmstart_path : Optional[str] = None,
    resume_path    : Optional[str] = None,
    log_dir        : str   = "./logs/ma",
    model_dir      : str   = "./models/ma",
    eval_freq      : int   = 25_000,
    checkpoint_freq: int   = 100_000,
    n_eval_games   : int   = 20,
    learning_rate  : float = 1e-4,   # lower than SA — shared policy is fragile
    n_steps        : int   = 2048,
    batch_size     : int   = 256,
    n_epochs       : int   = 10,
    verbose        : int   = 1,
):
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── Vectorised training envs ──────────────────────────────────────
    print(f"Creating {n_envs} parallel multi-agent environments...")
    vec_env = SubprocVecEnv([make_ma_env(i, seed) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(log_dir, "monitor"))

    # ── Model: resume, warm-start, or fresh ──────────────────────────
    policy_kwargs = dict(net_arch=[256, 256])

    if resume_path:
        print(f"Resuming from: {resume_path}")
        model = MaskablePPO.load(
            resume_path, env=vec_env,
            tensorboard_log=log_dir,
            verbose=verbose,
        )

    elif warmstart_path:
        # Load single-agent weights, then set env to MA vec env.
        # The policy architecture is identical, so weights transfer directly.
        print(f"Warm-starting from single-agent model: {warmstart_path}")
        model = MaskablePPO.load(
            warmstart_path, env=vec_env,
            tensorboard_log=log_dir,
            verbose=verbose,
            # Keep the same learning rate schedule but reset timestep counter
        )
        # Adjust LR for multi-agent stability
        model.learning_rate = learning_rate
        print(f"  Warm-start complete. Policy params: "
              f"{sum(p.numel() for p in model.policy.parameters()):,}")

    else:
        print("Training from scratch...")
        model = MaskablePPO(
            policy          = "MlpPolicy",
            env             = vec_env,
            learning_rate   = learning_rate,
            n_steps         = n_steps,
            batch_size      = batch_size,
            n_epochs        = n_epochs,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = log_dir,
            seed            = seed,
            verbose         = verbose,
        )

    print(f"Policy parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # ── Callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(checkpoint_freq // n_envs, 1),
        save_path   = model_dir,
        name_prefix = "somerset_ma",
    )
    log_cb = MATrainingCallback(log_freq=5, verbose=verbose)
    callbacks = CallbackList([checkpoint_cb, log_cb])

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nMA self-play training for {total_steps:,} steps...")
    print(f"Logs  → {log_dir}   (tensorboard --logdir {log_dir})")
    print(f"Models→ {model_dir}\n")

    model.learn(
        total_timesteps     = total_steps,
        callback            = callbacks,
        reset_num_timesteps = resume_path is None,
        progress_bar        = True,
    )

    # ── Save ──────────────────────────────────────────────────────────
    final_path = os.path.join(model_dir, "somerset_ma_final")
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}.zip")

    # ── Evaluate ──────────────────────────────────────────────────────
    print(f"\nEvaluating over {n_eval_games} full games...")
    stats = evaluate_ma(model, n_games=n_eval_games)
    print(f"  Win rate     : {stats['win_rate']*100:.1f}%  "
          f"(team 0 vs self-play)")
    print(f"  Score margin : {stats['mean_margin']:+.1f}")
    print(f"  Mean hands   : {stats['mean_hands']:.1f}")

    vec_env.close()
    return model, stats


# ═══════════════════════════════════════════════════════════════════════════════
# Watch a model play a full game
# ═══════════════════════════════════════════════════════════════════════════════

def play_game(model_path: str, n_games: int = 3, seed: int = 0):
    """Load a model and render full games to stdout."""
    model = MaskablePPO.load(model_path)

    for g in range(n_games):
        env = SomeRSetMAEnv(render_mode="human")
        env.reset(seed=seed + g)
        print(f"\n{'═'*55}")
        print(f"Game {g+1}")

        while env.agents:
            agent  = env.agent_selection
            obs    = env.observe(agent)
            mask   = env.action_mask(agent)
            action, _ = model.predict(
                obs[np.newaxis], action_masks=mask[np.newaxis],
                deterministic=True)
            env.step(int(action[0]))

        winner = env.game.winning_team
        scores = env.game.team_scores
        print(f"\n{'═'*55}")
        print(f"Game {g+1} result: Team {winner} wins | "
              f"Scores: T0={scores[0]:+d}  T1={scores[1]:+d} | "
              f"Hands: {env.game.hand_number}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-agent self-play training for Some-R-Set")
    parser.add_argument("--steps",      type=int,   default=1_000_000)
    parser.add_argument("--envs",       type=int,   default=4)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--warmstart",  type=str,   default=None,
        help="Path to single-agent model to warm-start from")
    parser.add_argument("--resume",     type=str,   default=None,
        help="Path to MA model to resume training")
    parser.add_argument("--play",       type=str,   default=None,
        help="Path to model to watch play")
    parser.add_argument("--eval-games", type=int,   default=20)
    parser.add_argument("--log-dir",    type=str,   default="./logs/ma")
    parser.add_argument("--model-dir",  type=str,   default="./models/ma")
    args = parser.parse_args()

    if args.play:
        play_game(args.play, n_games=3, seed=args.seed)
    else:
        train_ma(
            total_steps    = args.steps,
            n_envs         = args.envs,
            seed           = args.seed,
            warmstart_path = args.warmstart,
            resume_path    = args.resume,
            learning_rate  = args.lr,
            n_eval_games   = args.eval_games,
            log_dir        = args.log_dir,
            model_dir      = args.model_dir,
        )
