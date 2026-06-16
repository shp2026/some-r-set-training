"""
somerset_train.py
Trains a Some-R-Set agent using MaskablePPO from sb3-contrib.

Why MaskablePPO instead of plain PPO?
  Plain PPO samples actions from the full action space and then clips illegal
  ones — this wastes gradient signal and learns slowly.  MaskablePPO zero-out
  the logits of illegal actions *before* sampling, so the agent only ever
  considers valid moves.  This is essential for card games where the legal
  action set changes every single step.

File layout expected (all in the same directory):
  somerset_deck.py
  somerset_tricks.py
  somerset_scoring.py
  somerset_bidding.py
  somerset_env.py
  somerset_train.py   ← this file

Usage:
  python somerset_train.py              # train with defaults
  python somerset_train.py --steps 2_000_000 --envs 8
"""

import argparse
import os
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from somerset_env import SomeRSetEnv, PHASE_BIDDING, PHASE_DECLARING, PHASE_PLAYING

warnings.filterwarnings("ignore")


# ── 1. Action-mask wrapper ─────────────────────────────────────────────────────
# MaskablePPO needs a callable mask_fn(env) → bool array of shape (n_actions,)
# We attach the mask to the env via ActionMasker.

def get_mask(env: SomeRSetEnv) -> np.ndarray:
    """
    Return a boolean mask over the full action space.
    True  = this action is legal right now.
    False = illegal; MaskablePPO will never sample it.
    """
    legal = env.legal_actions   # set by the wrapper property below
    mask  = np.zeros(env.action_space.n, dtype=bool)
    for a in legal:
        mask[a] = True
    return mask


class SomeRSetMaskedEnv(SomeRSetEnv):
    """
    Thin subclass that:
      1. Exposes a `legal_actions` property (required by get_mask).
      2. Wraps info["legal_actions"] so ActionMasker can find it.
    """
    @property
    def legal_actions(self) -> list[int]:
        return self._legal_actions()

    def action_masks(self) -> np.ndarray:
        """Called directly by MaskablePPO at every step."""
        return get_mask(self)


# ── 2. Environment factory ─────────────────────────────────────────────────────

def make_env(rank: int, seed: int = 0):
    """Factory function for SubprocVecEnv — each worker gets its own env."""
    def _init():
        env = SomeRSetMaskedEnv(agent_player=0)
        env = Monitor(env)          # records episode rewards/lengths
        env.reset(seed=seed + rank)
        return env
    return _init


# ── 3. Custom training callback ────────────────────────────────────────────────

class SomeRSetCallback(BaseCallback):
    """
    Logs game-specific metrics to TensorBoard every N rollouts:
      - Mean episode reward
      - Win rate (agent's team reached 66 first)
      - Mean hands per game
      - Mean bid value
    """
    def __init__(self, log_freq: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq   = log_freq
        self.n_rollouts = 0

    def _on_rollout_end(self) -> bool:
        self.n_rollouts += 1
        if self.n_rollouts % self.log_freq != 0:
            return True

        # Pull episode info from the Monitor wrapper
        ep_infos = self.model.ep_info_buffer
        if not ep_infos:
            return True

        rewards = [ep["r"] for ep in ep_infos]
        lengths = [ep["l"] for ep in ep_infos]

        self.logger.record("somerset/mean_ep_reward", np.mean(rewards))
        self.logger.record("somerset/mean_ep_length", np.mean(lengths))

        if self.verbose >= 1:
            print(f"  Rollout {self.n_rollouts:4d} | "
                  f"mean_reward={np.mean(rewards):+.1f} | "
                  f"mean_ep_len={np.mean(lengths):.0f}")
        return True


# ── 4. Evaluation helper ───────────────────────────────────────────────────────

def evaluate(model: MaskablePPO, n_episodes: int = 50) -> dict:
    """
    Run n_episodes with the trained model (deterministic=True) and return stats.
    Returns dict with keys: mean_reward, win_rate, mean_hands.
    """
    env = SomeRSetMaskedEnv(agent_player=0)
    rewards, wins, hands = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        ep_reward = 0.0
        done      = False

        while not done:
            masks  = env.action_masks()
            action, _ = model.predict(
                obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        hands.append(env.game.hand_number)
        # Win = agent's team (0) reached 66 first
        wins.append(1 if env.game.winning_team == 0 else 0)

    return {
        "mean_reward" : np.mean(rewards),
        "std_reward"  : np.std(rewards),
        "win_rate"    : np.mean(wins),
        "mean_hands"  : np.mean(hands),
    }


# ── 5. Main training routine ───────────────────────────────────────────────────

def train(
    total_steps : int   = 500_000,
    n_envs      : int   = 4,
    seed        : int   = 42,
    log_dir     : str   = "./logs/somerset",
    model_dir   : str   = "./models/somerset",
    eval_freq   : int   = 20_000,
    checkpoint_freq: int = 50_000,
    n_eval_eps  : int   = 20,
    learning_rate: float = 3e-4,
    n_steps     : int   = 2048,    # rollout steps per env before update
    batch_size  : int   = 256,
    n_epochs    : int   = 10,
    gamma       : float = 0.99,
    verbose     : int   = 1,
):
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── Vectorised training envs ──────────────────────────────────────
    print(f"Creating {n_envs} parallel training environments...")
    vec_env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(log_dir, "monitor"))

    # ── Single eval env ───────────────────────────────────────────────
    eval_env_raw = SomeRSetMaskedEnv(agent_player=0)
    eval_env     = Monitor(eval_env_raw,
                           filename=os.path.join(log_dir, "eval_monitor"))

    # ── Model ─────────────────────────────────────────────────────────
    # Policy network: MlpPolicy with two hidden layers.
    # For card games, a larger network (256×256 or 512×512) often helps
    # because the agent must learn complex suit/trump interactions.
    policy_kwargs = dict(
        net_arch=[256, 256],   # two hidden layers of 256 units each
    )

    model = MaskablePPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = learning_rate,
        n_steps         = n_steps,
        batch_size      = batch_size,
        n_epochs        = n_epochs,
        gamma           = gamma,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,     # entropy bonus encourages exploration
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = policy_kwargs,
        tensorboard_log = log_dir,
        seed            = seed,
        verbose         = verbose,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"Observation size: {vec_env.observation_space.shape}")
    print(f"Action size:      {vec_env.action_space.n}")

    # ── Callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(checkpoint_freq // n_envs, 1),
        save_path   = model_dir,
        name_prefix = "somerset_ppo",
    )

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path = model_dir,
        log_path             = log_dir,
        eval_freq            = max(eval_freq // n_envs, 1),
        n_eval_episodes      = n_eval_eps,
        deterministic        = True,
        render               = False,
    )

    log_cb = SomeRSetCallback(log_freq=5, verbose=verbose)

    callbacks = CallbackList([checkpoint_cb, eval_cb, log_cb])

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nTraining for {total_steps:,} steps across {n_envs} envs...")
    print(f"Logs  → {log_dir}")
    print(f"Models→ {model_dir}")
    print(f"Run:  tensorboard --logdir {log_dir}\n")

    model.learn(
        total_timesteps     = total_steps,
        callback            = callbacks,
        reset_num_timesteps = True,
        progress_bar        = True,
    )

    # ── Save final model ──────────────────────────────────────────────
    final_path = os.path.join(model_dir, "somerset_ppo_final")
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}.zip")

    # ── Evaluate ──────────────────────────────────────────────────────
    print(f"\nEvaluating over {n_eval_eps} episodes...")
    stats = evaluate(model, n_episodes=n_eval_eps)
    print(f"  Mean reward : {stats['mean_reward']:+.1f} ± {stats['std_reward']:.1f}")
    print(f"  Win rate    : {stats['win_rate']*100:.1f}%")
    print(f"  Mean hands  : {stats['mean_hands']:.1f}")

    vec_env.close()
    eval_env.close()
    return model, stats


# ── 6. Load and play a saved model ────────────────────────────────────────────

def load_and_play(model_path: str, n_episodes: int = 5, render: bool = True):
    """Load a saved model and watch it play."""
    model = MaskablePPO.load(model_path)
    env   = SomeRSetMaskedEnv(
        agent_player=0,
        render_mode="human" if render else None,
    )

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        ep_reward = 0.0
        print(f"\n{'═'*55}")
        print(f"Episode {ep+1}")

        while True:
            masks  = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            if terminated or truncated:
                break

        result = "WIN" if env.game.winning_team == 0 else "LOSS"
        print(f"\n{result} | Reward: {ep_reward:+.1f} | "
              f"Hands: {env.game.hand_number} | "
              f"Final: {env.game.team_scores}")


# ── 7. CLI entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Some-R-Set PPO agent")
    parser.add_argument("--steps",    type=int,   default=500_000)
    parser.add_argument("--envs",     type=int,   default=4)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--eval",     type=int,   default=20_000,
                        help="Evaluate every N steps")
    parser.add_argument("--log-dir",  type=str,   default="./logs/somerset")
    parser.add_argument("--model-dir",type=str,   default="./models/somerset")
    parser.add_argument("--play",     type=str,   default=None,
                        help="Path to saved model to load and play")
    args = parser.parse_args()

    if args.play:
        load_and_play(args.play)
    else:
        train(
            total_steps = args.steps,
            n_envs      = args.envs,
            seed        = args.seed,
            learning_rate = args.lr,
            eval_freq   = args.eval,
            log_dir     = args.log_dir,
            model_dir   = args.model_dir,
        )
