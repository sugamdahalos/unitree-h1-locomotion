"""
scripts/evaluate.py: Evaluation and Trajectory Logger for Unitree H1.

Evaluates a trained policy across multiple episodes and saves
biomechanical time-series data to an .npz archive for analysis.

Features:
  - Transient Filtering: Rejects falling episodes from steady-state metrics.
  - Multi-Speed Testing: Supports evaluating policies at 0.5, 1.0, and 1.5 m/s.
  - Detailed Telemetry: Logs CoM, joint kinetics, ground contacts, and power.
"""

import os
import sys
import argparse
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import SAC
from configs.default import get_config, ExperimentConfig, EXPERIMENTS
from envs.unitree_h1_env import UnitreeH1Env
from src.utils import compute_mechanical_cot, compute_energy_exchange


def evaluate_single_episode(model, env, num_steps: int = 1000):
    """Run a single deterministic rollout and record all biomechanical signals."""
    log = {
        "time": [],
        "com_pos": [],
        "com_vel": [],
        "joint_pos": [],
        "joint_vel": [],
        "applied_torque": [],
        "actions": [],
        "rewards": [],
        "left_contact": [],
        "right_contact": [],
    }

    obs, _ = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False

    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        data = env.unwrapped.data
        log["time"].append(float(data.time))
        log["com_pos"].append(data.qpos[0:3].copy())
        log["com_vel"].append(data.qvel[0:3].copy())
        log["joint_pos"].append(data.qpos[env.unwrapped.policy_qpos_adrs].copy())
        log["joint_vel"].append(data.qvel[env.unwrapped.policy_qvel_adrs].copy())
        log["applied_torque"].append(env.unwrapped.applied_torque[env.unwrapped.policy_actuator_ids].copy())
        log["actions"].append(action.copy())
        log["rewards"].append(float(reward))
        log["left_contact"].append(info["contact"]["left"])
        log["right_contact"].append(info["contact"]["right"])

        total_reward += float(reward)
        obs = next_obs

        if terminated or truncated:
            break

    for k in log:
        log[k] = np.array(log[k])

    return log, total_reward, terminated


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Unitree H1 locomotion policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained policy checkpoint (.zip)")
    parser.add_argument("--exp", type=str, default="exp1", choices=list(EXPERIMENTS.keys()),
                        help=f"Experiment config to initialize environment with (choices: {list(EXPERIMENTS.keys())})")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes to collect")
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="Maximum steps per episode (default 1000 = 10.0s)")
    parser.add_argument("--target-vel", type=float, default=None,
                        help="Override target velocity during evaluation (e.g. 0.5, 1.0, 1.5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for evaluation data .npz archive")
    parser.add_argument("--device", type=str, default="auto",
                        help="PyTorch device")
    args = parser.parse_args()

    cfg = get_config(args.exp)
    if args.target_vel is not None:
        cfg.target_velocity = args.target_vel

    output_path = args.output or f"evaluation_data_{cfg.name}.npz"

    print("=" * 70)
    print(f"Evaluating Policy: {args.checkpoint}")
    print(f"Experiment Config: {cfg.name}")
    print(f"Target Velocity  : {cfg.target_velocity} m/s")
    print(f"Episodes         : {args.episodes}")
    print("=" * 70)

    env = UnitreeH1Env(config=cfg)
    model = SAC.load(args.checkpoint, device=args.device)

    all_logs = []
    rewards = []
    distances = []
    durations = []
    successes = []
    cots = []
    correlations = []

    for ep in range(args.episodes):
        log, ep_reward, fell = evaluate_single_episode(model, env, args.num_steps)
        ep_duration = len(log["time"]) * env.dt
        net_dist = float(log["com_pos"][-1, 0] - log["com_pos"][0, 0])
        avg_vel = float(np.mean(log["com_vel"][:, 0]))

        # An episode is considered steady-state walking only if it survives without falling
        # for at least 5.0 seconds (500 steps)
        is_steady = (not fell) and (len(log["time"]) >= 500)
        successes.append(is_steady)
        rewards.append(ep_reward)
        distances.append(net_dist)
        durations.append(ep_duration)

        if is_steady:
            # Discard first 1.0s (100 steps) startup transient for biomechanical metrics
            steady_start = 100
            steady_pos = log["com_pos"][steady_start:]
            steady_vel = log["com_vel"][steady_start:]
            steady_torques = log["applied_torque"][steady_start:]
            steady_qvel = log["joint_vel"][steady_start:]
            steady_dist = float(steady_pos[-1, 0] - steady_pos[0, 0])

            cot = compute_mechanical_cot(steady_torques, steady_qvel, steady_dist, dt=env.dt)
            _, _, _, r_corr = compute_energy_exchange(steady_pos, steady_vel)
            cots.append(cot)
            correlations.append(r_corr)
        else:
            cots.append(float("nan"))
            correlations.append(float("nan"))

        all_logs.append(log)

        status = "✓ COMPLETED" if is_steady else "✗ FELL / COLLAPSED"
        print(f"Episode {ep+1:2d}/{args.episodes}: Duration={ep_duration:5.2f}s | Dist={net_dist:5.2f}m | Vel={avg_vel:5.2f}m/s | Status={status}")

    env.close()

    # ── Summary Statistics ────────────────────────────────────────────
    valid_cots = [c for c in cots if not np.isnan(c)]
    valid_corrs = [r for r in correlations if not np.isnan(r)]
    success_rate = (sum(successes) / len(successes)) * 100.0

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Steady-State Success Rate : {success_rate:.1f}% ({sum(successes)}/{args.episodes})")
    print(f"Average Distance Traveled : {np.mean(distances):.2f} ± {np.std(distances):.2f} m")
    print(f"Average Episode Duration  : {np.mean(durations):.2f} ± {np.std(durations):.2f} s")
    if valid_cots:
        print(f"Mechanical Cost of Transport: {np.mean(valid_cots):.4f} ± {np.std(valid_cots):.4f}")
        print(f"KE/PE Phase Correlation     : {np.mean(valid_corrs):.4f} ± {np.std(valid_corrs):.4f}")
    else:
        print("Biomechanical Metrics: Insufficient sustained steady-state strides (robot fell).")
    print("=" * 70)

    # ── Save Best Representative Episode for Plotting ────────────────
    # Pick the longest / highest-reward episode
    best_idx = int(np.argmax(rewards))
    best_log = all_logs[best_idx]

    np.savez_compressed(
        output_path,
        time=best_log["time"],
        com_pos=best_log["com_pos"],
        com_vel=best_log["com_vel"],
        joint_pos=best_log["joint_pos"],
        joint_vel=best_log["joint_vel"],
        applied_torque=best_log["applied_torque"],
        actions=best_log["actions"],
        rewards=best_log["rewards"],
        left_contact=best_log["left_contact"],
        right_contact=best_log["right_contact"],
        summary_success_rate=success_rate,
        summary_avg_distance=np.mean(distances),
        summary_avg_duration=np.mean(durations),
        summary_mean_cot=np.mean(valid_cots) if valid_cots else -1.0,
        summary_mean_corr=np.mean(valid_corrs) if valid_corrs else 0.0,
    )
    print(f"Biomechanical data archive saved to: {output_path}")


if __name__ == "__main__":
    main()
