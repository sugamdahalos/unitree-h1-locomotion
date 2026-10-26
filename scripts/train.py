"""
scripts/train.py: SAC Training Script for Unitree H1 Locomotion.

Supports controlled ablation experiments:
  --exp exp1  : Naive baseline (stiff stance & knee-buckle collapse)
  --exp exp2  : Posture & effort regularized with alive bonus (frozen mannequin trap)
  --exp exp3  : Progress + effort minimization (zero alive bonus, scooter exploit)
  --exp exp4a : Exp 3 + bilateral symmetry penalty
  --exp exp4b : Exp 3 + swing foot clearance
  --exp exp4c : Exp 3 + contact alternation

Usage:
  # Quick smoke check (1,000 steps)
  python scripts/train.py --exp exp1 --smoke

  # Full research training run
  python scripts/train.py --exp exp2 --total-timesteps 3000000 --n-envs 16
"""

import os
import sys
import argparse
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from configs.default import get_config, ExperimentConfig, EXPERIMENTS
from envs.unitree_h1_env import UnitreeH1Env
from src.callbacks import VelocityCurriculumCallback, BiomechanicsTensorboardCallback


def make_env(config: ExperimentConfig, rank: int, seed: int = 42):
    """Factory function for creating single environment instances for vectorization."""
    def _init():
        env = UnitreeH1Env(config=config)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train SAC on Unitree H1 Humanoid")
    parser.add_argument("--exp", type=str, default="exp1", choices=list(EXPERIMENTS.keys()),
                        help=f"Experiment preset to train (choices: {list(EXPERIMENTS.keys())})")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override total timesteps (default: from config, 3M)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel environments (default: from config, 16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--target-vel", type=float, default=None,
                        help="Target forward velocity in m/s (default: 1.0 m/s)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable velocity curriculum (0.3 -> 1.0 m/s)")
    parser.add_argument("--device", type=str, default="auto",
                        help="PyTorch device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to checkpoint .zip to resume training from")
    parser.add_argument("--smoke", action="store_true",
                        help="Run quick sanity check (1,000 steps, single dummy env)")
    args = parser.parse_args()

    # ── Load Configuration ────────────────────────────────────────────
    cfg = get_config(args.exp)
    if args.target_vel is not None:
        cfg.target_velocity = args.target_vel
    if args.seed is not None:
        cfg.seed = args.seed

    total_timesteps = 1000 if args.smoke else (args.total_timesteps or cfg.total_timesteps)
    n_envs = 1 if args.smoke else (args.n_envs or cfg.n_envs)
    learning_starts = 100 if args.smoke else cfg.learning_starts

    print("=" * 70)
    print(f"Starting Unitree H1 Locomotion Training: {cfg.name.upper()}")
    print(f"Description     : {cfg.description}")
    print(f"Total Timesteps : {total_timesteps:,}")
    print(f"Parallel Envs   : {n_envs}")
    print(f"Target Velocity : {cfg.target_velocity} m/s")
    print(f"Device          : {args.device}")
    print(f"Curriculum      : {args.curriculum}")
    print("=" * 70)

    # ── Setup Logging & Checkpoints ───────────────────────────────────
    exp_checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.name)
    os.makedirs(exp_checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.tb_log_dir, exist_ok=True)

    # ── Create Vectorized Environments ────────────────────────────────
    env_fns = [make_env(cfg, rank=i, seed=cfg.seed) for i in range(n_envs)]
    if args.smoke or n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # Create separate evaluation environment
    eval_env = DummyVecEnv([make_env(cfg, rank=100, seed=cfg.seed + 100)])
    eval_env = VecMonitor(eval_env)

    # ── Configure Callbacks ───────────────────────────────────────────
    callbacks = []

    # Periodic checkpoint saver
    save_freq = 500 if args.smoke else max(1000, cfg.save_freq // n_envs)
    callbacks.append(CheckpointCallback(
        save_freq=save_freq,
        save_path=exp_checkpoint_dir,
        name_prefix=f"sac_h1_{cfg.name}",
        verbose=1,
    ))

    # Evaluation callback
    eval_freq = 500 if args.smoke else max(1000, cfg.eval_freq // n_envs)
    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=exp_checkpoint_dir,
        log_path=exp_checkpoint_dir,
        eval_freq=eval_freq,
        n_eval_episodes=2 if args.smoke else cfg.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    ))

    # Detailed biomechanics telemetry logger for TensorBoard
    callbacks.append(BiomechanicsTensorboardCallback(log_freq=100 if args.smoke else 1000))

    # Optional velocity curriculum
    if args.curriculum:
        callbacks.append(VelocityCurriculumCallback(
            initial_vel=0.3,
            target_vel=cfg.target_velocity,
            ramp_timesteps=min(500_000, total_timesteps // 2),
        ))

    callback_list = CallbackList(callbacks)

    # ── Initialize SAC Model ──────────────────────────────────────────
    policy_kwargs = dict(net_arch=dict(pi=cfg.net_arch, qf=cfg.net_arch))

    if args.load:
        print(f"Resuming training from checkpoint: {args.load}")
        model = SAC.load(
            args.load,
            env=vec_env,
            device=args.device,
            tensorboard_log=cfg.tb_log_dir,
        )
    else:
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=learning_starts,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            gamma=cfg.gamma,
            train_freq=cfg.train_freq,
            gradient_steps=cfg.gradient_steps,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=cfg.seed,
            device=args.device,
            tensorboard_log=cfg.tb_log_dir,
        )

    # ── Train Policy ──────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            tb_log_name=f"SAC_H1_{cfg.name.upper()}",
            reset_num_timesteps=args.load is None,
        )
        final_save_path = os.path.join(exp_checkpoint_dir, "final_model.zip")
        model.save(final_save_path)
        print(f"\nTraining completed successfully! Final model saved to: {final_save_path}")
    finally:
        vec_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
