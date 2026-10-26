"""
scripts/run_overnight.py: Sequential Overnight Experiment Runner.

Executes a sequence of research experiments uninterrupted:
  1. Train policy (SAC, 3M timesteps, 16 parallel envs)
  2. Evaluate policy (20 randomized test episodes, logs telemetry to .npz)
  3. Generate biomechanical analysis plot (.png)
  4. Advance to next experiment automatically

Features:
  - Fault Tolerance: Errors in one experiment are logged; the suite continues to the next.
  - Live Consolidated Logging: Streams to stdout and appends to 'overnight_run.log'.
  - Summary Markdown Table: Compiles a side-by-side comparative table in 'overnight_summary.md'.

Usage:
  # Quick dry-run / smoke test (500 steps per experiment)
  python scripts/run_overnight.py --smoke

  # Full research overnight run (exp4a, exp4b, exp4c)
  python scripts/run_overnight.py

  # Custom experiment list
  python scripts/run_overnight.py --experiments exp4a exp4b exp4c
"""

import os
import sys
import time
import argparse
import subprocess
import datetime
from typing import List, Dict, Any
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default import get_config, EXPERIMENTS


LOG_FILE = os.path.join(PROJECT_ROOT, "overnight_run.log")
SUMMARY_FILE = os.path.join(PROJECT_ROOT, "overnight_summary.md")


def log_msg(msg: str):
    """Write timestamped message to stdout and append to overnight_run.log."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


def run_command_live(cmd: List[str], description: str) -> bool:
    """Execute a subprocess while streaming stdout/stderr to console and log."""
    log_msg(f"STARTING: {description}")
    log_msg(f"COMMAND : {' '.join(cmd)}")
    start_time = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT,
        )

        with open(LOG_FILE, "a") as f:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)

        proc.wait()
        elapsed = time.time() - start_time
        mins = elapsed / 60.0

        if proc.returncode == 0:
            log_msg(f"SUCCESS: {description} completed in {mins:.1f} minutes.\n")
            return True
        else:
            log_msg(f"ERROR: {description} failed with return code {proc.returncode} after {mins:.1f} minutes.\n")
            return False

    except Exception as e:
        log_msg(f"EXCEPTION during {description}: {str(e)}\n")
        return False


def run_single_experiment(
    exp_name: str,
    python_bin: str,
    smoke: bool = False,
    total_timesteps: int = None,
    n_envs: int = None,
    eval_episodes: int = 20,
) -> Dict[str, Any]:
    """Execute the full Train -> Evaluate -> Analyze pipeline for one experiment."""
    cfg = get_config(exp_name)
    actual_exp = cfg.name
    res = {
        "exp": exp_name,
        "name": actual_exp,
        "train_success": False,
        "eval_success": False,
        "analysis_success": False,
    }

    log_msg("=" * 80)
    log_msg(f"STARTING PIPELINE FOR EXPERIMENT: {exp_name.upper()} ({actual_exp})")
    log_msg(f"Description: {cfg.description}")
    log_msg("=" * 80)

    # ── STEP 1: TRAIN ─────────────────────────────────────────────────
    train_cmd = [python_bin, "scripts/train.py", "--exp", exp_name]
    if smoke:
        train_cmd.append("--smoke")
    else:
        if total_timesteps is not None:
            train_cmd.extend(["--total-timesteps", str(total_timesteps)])
        if n_envs is not None:
            train_cmd.extend(["--n-envs", str(n_envs)])

    train_ok = run_command_live(train_cmd, f"Training {exp_name}")
    res["train_success"] = train_ok
    if not train_ok:
        log_msg(f"Skipping evaluation for {exp_name} due to training failure.")
        return res

    # ── STEP 2: EVALUATE ──────────────────────────────────────────────
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", actual_exp, "best_model.zip")
    if not os.path.exists(checkpoint_path):
        # Fallback to final model if best_model wasn't written
        checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", actual_exp, "final_model.zip")

    if not os.path.exists(checkpoint_path):
        log_msg(f"ERROR: Checkpoint not found at {checkpoint_path}. Skipping evaluation.")
        return res

    eval_data_path = f"evaluation_data_{exp_name}.npz"
    episodes = 2 if smoke else eval_episodes
    eval_cmd = [
        python_bin,
        "scripts/evaluate.py",
        "--checkpoint", checkpoint_path,
        "--exp", exp_name,
        "--episodes", str(episodes),
        "--output", eval_data_path,
    ]
    eval_ok = run_command_live(eval_cmd, f"Evaluating {exp_name}")
    res["eval_success"] = eval_ok
    if not eval_ok:
        log_msg(f"Skipping biomechanical plotting for {exp_name} due to eval failure.")
        return res

    # ── STEP 3: ANALYZE BIOMECHANICS & PLOT ────────────────────────────
    plot_path = f"{exp_name}_biomechanical_analysis.png"
    analyze_cmd = [
        python_bin,
        "scripts/analyze_biomechanics.py",
        "--data", eval_data_path,
        "--output", plot_path,
    ]
    analyze_ok = run_command_live(analyze_cmd, f"Biomechanical Analysis for {exp_name}")
    res["analysis_success"] = analyze_ok

    return res


def generate_morning_summary(results: List[Dict[str, Any]]):
    """Compile a Markdown summary table of all completed experiments."""
    lines = [
        "# Overnight Locomotion Experiment Summary",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## 1. Pipeline Execution Status",
        "| Experiment | Config Name | Training | Evaluation | Biomechanical Plot |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]

    for r in results:
        t_icon = "PASS" if r["train_success"] else "FAIL"
        e_icon = "PASS" if r["eval_success"] else "FAIL"
        a_icon = "PASS" if r["analysis_success"] else "FAIL"
        lines.append(f"| `{r['exp']}` | `{r['name']}` | {t_icon} | {e_icon} | {a_icon} |")

    lines.append("\n## 2. Quantitative Biomechanical Comparison\n")
    lines.append("| Experiment | Success Rate | Distance (m) | Duration (s) | Cost of Transport (CoT) | KE/PE Corr (r) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        exp_name = r["exp"]
        data_file = os.path.join(PROJECT_ROOT, f"evaluation_data_{exp_name}.npz")
        if os.path.exists(data_file):
            try:
                d = np.load(data_file)
                succ = f"{float(d['summary_success_rate']):.1f}%" if "summary_success_rate" in d else "N/A"
                dist = f"{float(d['summary_avg_distance']):.2f}m" if "summary_avg_distance" in d else "N/A"
                dur = f"{float(d['summary_avg_duration']):.2f}s" if "summary_avg_duration" in d else "N/A"
                cot_val = float(d['summary_mean_cot']) if "summary_mean_cot" in d else -1.0
                cot = f"{cot_val:.3f}" if cot_val > 0 else "N/A"
                corr_val = float(d['summary_mean_corr']) if "summary_mean_corr" in d else 0.0
                corr = f"{corr_val:.3f}" if "summary_mean_corr" in d else "N/A"
                lines.append(f"| **{exp_name}** | {succ} | {dist} | {dur} | {cot} | {corr} |")
            except Exception as e:
                lines.append(f"| **{exp_name}** | Error loading data: {e} | N/A | N/A | N/A | N/A |")
        else:
            lines.append(f"| **{exp_name}** | (No data file) | N/A | N/A | N/A | N/A |")

    summary_text = "\n".join(lines) + "\n"
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary_text)

    log_msg("=" * 80)
    log_msg("OVERNIGHT SUMMARY COMPILED:")
    print(summary_text)
    log_msg(f"Summary saved to {SUMMARY_FILE}")
    log_msg("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Automated Sequential Overnight Experiment Suite")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["exp4a", "exp4b", "exp4c"],
        help="List of experiment names to run sequentially (default: exp4a exp4b exp4c)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=3_000_000,
        help="Total timesteps per experiment (default: 3,000,000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=12,
        help="Number of parallel environments (default: 12)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test for all experiments (500 steps each) to verify pipeline",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable to invoke (default: current interpreter)",
    )
    args = parser.parse_args()

    suite_start = time.time()
    log_msg(f"OVERNIGHT EXPERIMENT RUNNER INITIALIZED AT {datetime.datetime.now()}")
    log_msg(f"Target Experiments : {args.experiments}")
    log_msg(f"Steps per Exp      : {args.total_timesteps:,} (Smoke: {args.smoke})")
    log_msg(f"Parallel Envs      : {args.n_envs}")
    log_msg(f"Python Executable  : {args.python_bin}")
    log_msg(f"Log File           : {LOG_FILE}")
    log_msg(f"Summary File       : {SUMMARY_FILE}")

    results = []
    for exp in args.experiments:
        try:
            res = run_single_experiment(
                exp_name=exp,
                python_bin=args.python_bin,
                smoke=args.smoke,
                total_timesteps=args.total_timesteps,
                n_envs=args.n_envs,
                eval_episodes=args.eval_episodes,
            )
            results.append(res)
        except Exception as e:
            log_msg(f"UNCAUGHT FATAL ERROR on experiment {exp}: {e}")
            results.append({
                "exp": exp,
                "name": exp,
                "train_success": False,
                "eval_success": False,
                "analysis_success": False,
            })

    # Generate consolidated report
    generate_morning_summary(results)

    total_hours = (time.time() - suite_start) / 3600.0
    log_msg(f"ALL EXPERIMENTS FINISHED IN {total_hours:.2f} HOURS.")


if __name__ == "__main__":
    main()
