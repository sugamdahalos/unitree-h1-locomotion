"""
scripts/watch.py: Interactive MuJoCo Viewer for Trained Unitree H1 Policies.

Features:
  - Real-time physics synchronization with slow-motion support.
  - Automatic 3D camera tracking that follows the robot's pelvis.
  - Live console telemetry displaying velocity, height, and applied torques.
"""

import os
import sys
import time
import argparse
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mujoco
import mujoco.viewer
from stable_baselines3 import SAC

from configs.default import get_config, EXPERIMENTS
from envs.unitree_h1_env import UnitreeH1Env


def main():
    parser = argparse.ArgumentParser(description="Watch trained Unitree H1 locomotion policy in MuJoCo viewer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model .zip checkpoint")
    parser.add_argument("--exp", type=str, default="exp1", choices=list(EXPERIMENTS.keys()),
                        help=f"Experiment config name (choices: {list(EXPERIMENTS.keys())})")
    parser.add_argument("--target-vel", type=float, default=None,
                        help="Target forward velocity command in m/s")
    parser.add_argument("--slowdown", type=float, default=1.0,
                        help="Playback slowdown factor (1.0 = real-time, 2.0 = half-speed)")
    args = parser.parse_args()

    cfg = get_config(args.exp)
    if args.target_vel is not None:
        cfg.target_velocity = args.target_vel

    print("=" * 60)
    print(f"Loading checkpoint : {args.checkpoint}")
    print(f"Experiment preset  : {cfg.name}")
    print(f"Target velocity    : {cfg.target_velocity} m/s")
    print("Press [Space] in MuJoCo viewer to pause/resume.")
    print("=" * 60)

    env = UnitreeH1Env(config=cfg)
    model = SAC.load(args.checkpoint, device="auto")

    obs, _ = env.reset()

    # Launch passive interactive viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Configure tracking camera to follow pelvis
        pelvis_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_body_id != -1:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis_body_id
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -15.0

        step = 0
        while viewer.is_running():
            step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Keep camera synced
            viewer.sync()

            step += 1
            if step % 25 == 0:
                tel = info["telemetry"]
                sys.stdout.write(
                    f"\rStep: {step:4d} | Height: {tel['com_height']:.3f}m | "
                    f"Vel: {tel['forward_velocity']:+.2f} m/s | Dist: {tel['distance_traveled']:.2f}m | "
                    f"Rew: {reward:+.2f}"
                )
                sys.stdout.flush()

            if terminated or truncated:
                print(f"\nEpisode finished ({'Fell' if terminated else 'Max steps'}). Resetting...")
                obs, _ = env.reset()
                step = 0
                time.sleep(0.5)

            # Sleep to match real-time playback
            elapsed = time.time() - step_start
            sleep_duration = (env.dt * args.slowdown) - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    env.close()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
