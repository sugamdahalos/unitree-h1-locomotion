"""
scripts/record_video.py: Offscreen Video Recorder for Unitree H1 Locomotion.

Renders high-definition MP4 videos of trained humanoid locomotion policies
in MuJoCo using an automated 3D tracking camera and direct ffmpeg pipe encoding.

Features:
  - No GUI display required (works fully headless/offscreen via MuJoCo EGL/OSMesa).
  - High-Definition Output: 1280x720p H.264 MP4 with optimal compression.
  - Smooth 3D Tracking Camera: Follows the robot's pelvis in real-time.
  - Camera Perspectives: 'diagonal' (3/4 view), 'side' (profile view), 'follow' (chase cam).
  - Batch Mode (--all): Records videos for all available experiment checkpoints in one pass.

Usage:
  # Record a single experiment (e.g. Exp 3)
  python scripts/record_video.py --exp exp3

  # Record with custom camera view and 2x slow-motion
  python scripts/record_video.py --exp exp4a --camera side --slowdown 2.0

  # Batch record all experiments (Exp 1 through Exp 4C)
  python scripts/record_video.py --all
"""

import os
import sys
import argparse
import subprocess
from typing import Optional, Dict
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mujoco
from stable_baselines3 import SAC
from configs.default import get_config, EXPERIMENTS
from envs.unitree_h1_env import UnitreeH1Env


CAMERA_PRESETS: Dict[str, Dict[str, float]] = {
    "diagonal": {"distance": 3.2, "elevation": -12.0, "azimuth": 135.0},
    "side":     {"distance": 3.2, "elevation": -10.0, "azimuth": 90.0},
    "follow":   {"distance": 3.0, "elevation": -15.0, "azimuth": 180.0},
    "front":    {"distance": 3.0, "elevation": -10.0, "azimuth": 0.0},
}


def record_policy_video(
    checkpoint_path: str,
    exp_name: str,
    output_path: str,
    max_steps: int = 1000,
    width: int = 1280,
    height: int = 720,
    fps: int = 50,
    camera_view: str = "diagonal",
    slowdown: float = 1.0,
    device: str = "auto",
) -> bool:
    """Run deterministic rollout and render frames directly into an MP4 container."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' does not exist.")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cfg = get_config(exp_name)

    print("=" * 70)
    print(f"Recording Video: {exp_name.upper()} ({cfg.name})")
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"Output File    : {output_path}")
    print(f"Resolution     : {width}x{height} @ {fps} fps (slowdown={slowdown}x)")
    print(f"Camera View    : {camera_view}")
    print("=" * 70)

    # Initialize Environment & Model
    env = UnitreeH1Env(config=cfg)
    model = SAC.load(checkpoint_path, device=device)

    # Initialize MuJoCo Offscreen Renderer
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    # Configure Tracking Camera
    cam = mujoco.MjvCamera()
    pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id != -1:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = pelvis_id
        preset = CAMERA_PRESETS.get(camera_view, CAMERA_PRESETS["diagonal"])
        cam.distance = preset["distance"]
        cam.elevation = preset["elevation"]
        cam.azimuth = preset["azimuth"]

    # Calculate frame recording interval
    # env.dt = 0.01s (100 Hz). At 50 fps real-time, record every (100 / 50) = 2 steps.
    # With slowdown factor S, step_interval = round(2 / S).
    step_interval = max(1, int(round((1.0 / (env.dt * fps)) / slowdown)))

    # Launch ffmpeg subprocess pipe
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        output_path,
    ]

    try:
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("Error: ffmpeg binary not found on system PATH. Please install ffmpeg.")
        env.close()
        return False

    obs, _ = env.reset(seed=42)
    step_count = 0
    frames_written = 0

    print("Rendering frames to MP4...")
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if step_count % step_interval == 0:
            renderer.update_scene(env.data, camera=cam)
            frame = renderer.render()
            proc.stdin.write(frame.tobytes())
            frames_written += 1

        if terminated or truncated:
            # Add 0.5s pause on final fallen/finished frame
            freeze_frames = int(fps * 0.5)
            renderer.update_scene(env.data, camera=cam)
            last_frame = renderer.render().tobytes()
            for _ in range(freeze_frames):
                proc.stdin.write(last_frame)
                frames_written += 1
            break

    proc.stdin.close()
    proc.wait()
    env.close()

    if proc.returncode == 0:
        duration_s = frames_written / fps
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"SUCCESS: Video saved to {output_path} ({duration_s:.1f}s, {file_size_mb:.2f} MB)")
        return True
    else:
        err = proc.stderr.read().decode()
        print(f"ERROR: ffmpeg failed with return code {proc.returncode}:\n{err}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Record offscreen MuJoCo MP4 videos of trained policies")
    parser.add_argument("--exp", type=str, default="exp3", choices=list(EXPERIMENTS.keys()),
                        help="Experiment config name")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model .zip (defaults to checkpoints/<exp_name>/best_model.zip)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MP4 filepath (default: videos/<exp_name>.mp4)")
    parser.add_argument("--camera", type=str, default="diagonal", choices=list(CAMERA_PRESETS.keys()),
                        help="Camera angle ('diagonal', 'side', 'follow', 'front')")
    parser.add_argument("--slowdown", type=float, default=1.0,
                        help="Playback slowdown factor (1.0 = real-time, 2.0 = 2x slow-mo)")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels")
    parser.add_argument("--fps", type=int, default=50, help="Video frame rate")
    parser.add_argument("--all", action="store_true",
                        help="Batch record MP4 videos for all experiments (exp1, exp2, exp3, exp4a, exp4b, exp4c)")
    args = parser.parse_args()

    if args.all:
        target_experiments = ["exp1", "exp2", "exp3", "exp4a", "exp4b", "exp4c"]
        print(f"Batch recording videos for {len(target_experiments)} experiments...")
        success_count = 0
        for exp in target_experiments:
            cfg = get_config(exp)
            ckpt = os.path.join(PROJECT_ROOT, "checkpoints", cfg.name, "best_model.zip")
            if not os.path.exists(ckpt):
                ckpt = os.path.join(PROJECT_ROOT, "checkpoints", cfg.name, "final_model.zip")

            out = os.path.join(PROJECT_ROOT, "videos", f"{cfg.name}.mp4")
            ok = record_policy_video(
                checkpoint_path=ckpt,
                exp_name=exp,
                output_path=out,
                camera_view=args.camera,
                slowdown=args.slowdown,
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
            if ok:
                success_count += 1
        print(f"\nBatch recording complete: {success_count}/{len(target_experiments)} videos saved to 'videos/' directory.")
    else:
        cfg = get_config(args.exp)
        ckpt = args.checkpoint or os.path.join(PROJECT_ROOT, "checkpoints", cfg.name, "best_model.zip")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(PROJECT_ROOT, "checkpoints", cfg.name, "final_model.zip")

        out = args.output or os.path.join(PROJECT_ROOT, "videos", f"{cfg.name}.mp4")
        record_policy_video(
            checkpoint_path=ckpt,
            exp_name=args.exp,
            output_path=out,
            camera_view=args.camera,
            slowdown=args.slowdown,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
