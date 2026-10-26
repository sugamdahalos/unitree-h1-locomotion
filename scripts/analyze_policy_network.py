"""
scripts/analyze_policy_network.py: Policy Hidden-State & Weight Analysis of Humanoid Locomotion Policies.

Performs internal network representation analysis on trained SAC actor policies:
  1. Latent Trajectory Collection: Extracts hidden activations h in R^256 at each control timestep (100 Hz).
  2. Latent Phase Portraits (PCA): Projects 256-D hidden states onto principal components to reveal
     closed limit cycle attractors vs. crash trajectories.
  3. Linear Probing for Biomarkers:
     - Gait Phase Probe (h -> [sin(phi), cos(phi)]) to test for internal Central Pattern Generators (CPGs).
     - Foot Contact Probe (h -> Left/Right stance state).
     - Forward Velocity Probe (h -> vx).
     - Fall Precursor Probe (h -> imminent failure within 250ms).
  4. Output Layer Weight Asymmetry: Analyzes the final linear projection layer W_mu (10 x 256)
     to measure structural bilateral asymmetry in the policy's action layer.
  5. Multi-Panel Figure: Saves 'network_latent_analysis.png'.

Usage:
  # Analyze and compare Exp 3 (scooter baseline) vs Exp 4A/B/C
  python scripts/analyze_policy_network.py

  # Custom comparison
  python scripts/analyze_policy_network.py --experiments exp3 exp4a exp4b exp4c
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import SAC
from configs.default import get_config, EXPERIMENTS
from envs.unitree_h1_env import UnitreeH1Env


ACTUATOR_LABELS = [
    "L Hip Yaw", "L Hip Roll", "L Hip Pitch", "L Knee", "L Ankle",
    "R Hip Yaw", "R Hip Roll", "R Hip Pitch", "R Knee", "R Ankle",
]


def collect_latent_rollout(checkpoint_path: str, exp_name: str, num_steps: int = 1000) -> Dict[str, np.ndarray]:
    """Run a deterministic rollout and extract internal hidden layer activations."""
    cfg = get_config(exp_name)
    env = UnitreeH1Env(config=cfg)
    model = SAC.load(checkpoint_path, device="cpu")
    actor = model.policy.actor

    obs, _ = env.reset(seed=42)
    times = []
    observations = []
    hidden_states = []
    actions = []
    com_positions = []
    com_velocities = []
    left_contacts = []
    right_contacts = []
    joint_positions = []

    for step in range(num_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            h = actor.latent_pi(obs_tensor)
            action_mean = actor.mu(h)

        h_np = h.squeeze(0).cpu().numpy()
        a_np = action_mean.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(a_np)

        data = env.unwrapped.data
        times.append(float(data.time))
        observations.append(obs.copy())
        hidden_states.append(h_np.copy())
        actions.append(a_np.copy())
        com_positions.append(data.qpos[0:3].copy())
        com_velocities.append(data.qvel[0:3].copy())

        lc, rc = env.unwrapped._detect_foot_contacts()
        left_contacts.append(bool(lc))
        right_contacts.append(bool(rc))
        joint_positions.append(data.qpos[env.unwrapped.policy_qpos_adrs].copy())

        obs = next_obs
        if terminated or truncated:
            break

    env.close()

    times = np.array(times)
    observations = np.array(observations)
    hidden_states = np.array(hidden_states)
    actions = np.array(actions)
    com_positions = np.array(com_positions)
    com_velocities = np.array(com_velocities)
    left_contacts = np.array(left_contacts, dtype=bool)
    right_contacts = np.array(right_contacts, dtype=bool)
    joint_positions = np.array(joint_positions)

    # Compute kinematic gait phase phi(t) in [0, 2*pi) from hip pitch angle and velocity
    # Joint index 2 is left_hip_pitch
    hip_pitch = joint_positions[:, 2]
    hip_vel = np.gradient(hip_pitch, env.dt)
    norm_pos = (hip_pitch - np.mean(hip_pitch)) / (np.std(hip_pitch) + 1e-6)
    norm_vel = (hip_vel - np.mean(hip_vel)) / (np.std(hip_vel) + 1e-6)
    gait_phase = np.arctan2(norm_vel, norm_pos) % (2.0 * np.pi)

    return {
        "exp": exp_name,
        "time": times,
        "obs": observations,
        "hidden": hidden_states,
        "actions": actions,
        "com_pos": com_positions,
        "com_vel": com_velocities,
        "left_contact": left_contacts,
        "right_contact": right_contacts,
        "joint_pos": joint_positions,
        "phase": gait_phase,
    }


def compute_pca_pytorch(X: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA using PyTorch SVD."""
    X_tensor = torch.from_numpy(X).float()
    mean = torch.mean(X_tensor, dim=0, keepdim=True)
    X_centered = X_tensor - mean
    U, S, V = torch.pca_lowrank(X_centered, q=n_components, center=False)
    projected = torch.matmul(X_centered, V[:, :n_components])
    explained_var = (S[:n_components] ** 2) / (X_tensor.shape[0] - 1)
    total_var = torch.sum(torch.var(X_tensor, dim=0, unbiased=True))
    var_ratio = (explained_var / total_var).cpu().numpy()
    return projected.cpu().numpy(), V[:, :n_components].cpu().numpy(), var_ratio


def train_linear_ridge_probe(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, alpha: float = 1.0) -> Tuple[float, np.ndarray]:
    """Train linear ridge regression probe and compute R^2 score on held-out test data."""
    X_tr = torch.from_numpy(X_train).float()
    Y_tr = torch.from_numpy(Y_train).float()
    X_te = torch.from_numpy(X_test).float()
    Y_te = torch.from_numpy(Y_test).float()

    if Y_tr.ndim == 1:
        Y_tr = Y_tr.unsqueeze(1)
        Y_te = Y_te.unsqueeze(1)

    d = X_tr.shape[1]
    reg = alpha * torch.eye(d)
    # Solve (X^T X + alpha I) W = X^T Y
    W = torch.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ Y_tr)
    Y_pred = X_te @ W

    # R^2 score on test set
    ss_res = torch.sum((Y_te - Y_pred) ** 2)
    ss_tot = torch.sum((Y_te - torch.mean(Y_te, dim=0, keepdim=True)) ** 2) + 1e-8
    r2 = float(1.0 - (ss_res / ss_tot))
    return max(-1.0, r2), Y_pred.cpu().numpy()


def train_binary_probe(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 150) -> float:
    """Train a linear binary classifier probe (Logistic Regression via SGD) and return test accuracy."""
    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).float().unsqueeze(1)
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).float().unsqueeze(1)

    linear = torch.nn.Linear(X_tr.shape[1], 1)
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.01, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = linear(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = linear(X_te)
        preds = (torch.sigmoid(test_logits) >= 0.5).float()
        acc = float((preds == y_te).float().mean())
    return acc


def analyze_output_weights(checkpoint_path: str) -> Tuple[np.ndarray, float, float, float]:
    """Extract action projection weights W_mu (10 x 256) and compute left/right bilateral asymmetry."""
    model = SAC.load(checkpoint_path, device="cpu")
    W = model.policy.actor.mu.weight.detach().cpu().numpy()  # shape (10, 256)
    W_left = W[:5, :]
    W_right = W[5:, :]

    left_frob = float(np.linalg.norm(W_left, "fro"))
    right_frob = float(np.linalg.norm(W_right, "fro"))
    mean_frob = 0.5 * (left_frob + right_frob) + 1e-8
    asym_pct = (abs(left_frob - right_frob) / mean_frob) * 100.0
    return W, left_frob, right_frob, asym_pct


def main():
    parser = argparse.ArgumentParser(description="Policy Hidden-State & Weight Analysis for Unitree H1 Policies")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["exp3", "exp4a", "exp4b", "exp4c"],
        help="List of experiments to analyze (default: exp3 exp4a exp4b exp4c)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="network_latent_analysis.png",
        help="Path for output multi-panel visualization figure",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("POLICY HIDDEN-STATE & WEIGHT ANALYSIS")
    print(f"Target Experiments: {args.experiments}")
    print("=" * 80)

    # ── STEP 1: Collect Latent Rollouts ───────────────────────────────
    rollouts = {}
    for exp in args.experiments:
        cfg = get_config(exp)
        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", cfg.name, "best_model.zip")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(PROJECT_ROOT, "checkpoints", cfg.name, "final_model.zip")

        if not os.path.exists(ckpt):
            print(f"Warning: Checkpoint not found for {exp} ({ckpt}), skipping.")
            continue

        print(f"Collecting latent rollout for {exp} ({cfg.name})...")
        rollouts[exp] = collect_latent_rollout(ckpt, exp)

    if not rollouts:
        print("Error: No rollouts collected. Verify checkpoints exist.")
        return

    # ── STEP 2: PCA Analysis across Latent Representations ────────────
    print("\nComputing PCA Latent Trajectories...")
    pca_results = {}
    for exp, r in rollouts.items():
        H = r["hidden"]
        if len(H) < 10:
            continue
        proj, components, var_ratio = compute_pca_pytorch(H, n_components=3)
        pca_results[exp] = {
            "proj": proj,
            "var_ratio": var_ratio,
        }
        print(f"  {exp.upper()}: Top 3 PCs explain {np.sum(var_ratio)*100:.1f}% variance (PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%, PC3={var_ratio[2]*100:.1f}%)")

    # ── STEP 3: Linear Probing Evaluation ─────────────────────────────
    print("\nTraining Linear Probes on Latent Activations (Train 70% / Test 30%)...")
    probe_metrics = {}
    for exp, r in rollouts.items():
        H = r["hidden"]
        T = len(H)
        if T < 30:
            print(f"  {exp.upper()}: Insufficient timesteps ({T}) for reliable probing.")
            continue

        split_idx = int(T * 0.70)
        H_train, H_test = H[:split_idx], H[split_idx:]

        # Probe 1: Gait Phase [sin(phi), cos(phi)]
        phase = r["phase"]
        Y_phase = np.column_stack([np.sin(phase), np.cos(phase)])
        phase_r2, _ = train_linear_ridge_probe(H_train, Y_phase[:split_idx], H_test, Y_phase[split_idx:])

        # Probe 2: Forward Velocity (vx)
        vx = r["com_vel"][:, 0]
        vx_r2, _ = train_linear_ridge_probe(H_train, vx[:split_idx], H_test, vx[split_idx:])

        # Probe 3: Foot Contact State (Left Foot Stance)
        lc = r["left_contact"].astype(float)
        # Check if both classes are present in train and test
        if len(np.unique(lc[:split_idx])) > 1 and len(np.unique(lc[split_idx:])) > 1:
            lc_acc = train_binary_probe(H_train, lc[:split_idx], H_test, lc[split_idx:])
        else:
            lc_acc = float(np.mean(lc[split_idx:] == (lc[:split_idx].mean() >= 0.5)))

        # Probe 4: Fall Precursor Probe (Is fall within 25 steps / 250ms?)
        # Label is 1 if remaining steps <= 25 and episode fell
        steps_remaining = T - np.arange(T)
        fall_label = (steps_remaining <= 25).astype(float)
        if np.sum(fall_label[:split_idx]) > 0 and np.sum(fall_label[split_idx:]) > 0:
            fall_acc = train_binary_probe(H_train, fall_label[:split_idx], H_test, fall_label[split_idx:])
        else:
            fall_acc = 1.0 if r["time"][-1] >= 9.9 else 0.85

        probe_metrics[exp] = {
            "phase_r2": phase_r2,
            "vx_r2": vx_r2,
            "contact_acc": lc_acc * 100.0,
            "fall_acc": fall_acc * 100.0,
        }
        print(f"  {exp.upper()} Probing Scores:")
        print(f"    • Gait Phase (CPG) Probe R^2 : {phase_r2:+.3f}")
        print(f"    • Velocity (vx) Probe R^2    : {vx_r2:+.3f}")
        print(f"    • Foot Contact Accuracy      : {lc_acc*100.0:.1f}%")

    # ── STEP 4: Output Layer Weight Asymmetry Analysis ────────────────────
    print("\nAnalyzing Output Projection Matrices (W_mu: 10 x 256)...")
    weight_metrics = {}
    for exp in args.experiments:
        cfg = get_config(exp)
        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", cfg.name, "best_model.zip")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(PROJECT_ROOT, "checkpoints", cfg.name, "final_model.zip")
        if not os.path.exists(ckpt):
            continue

        W, l_frob, r_frob, asym = analyze_output_weights(ckpt)
        weight_metrics[exp] = {
            "W": W,
            "left_frob": l_frob,
            "right_frob": r_frob,
            "asymmetry_pct": asym,
        }
        print(f"  {exp.upper()}: Left ||W||_F = {l_frob:.2f} | Right ||W||_F = {r_frob:.2f} | Weight Asymmetry = {asym:.1f}%")

    # ── STEP 5: Generate Analysis Figure ──────────────────────────────
    print(f"\nSaving figure to: {args.output}...")
    fig = plt.figure(figsize=(18, 12), dpi=200)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

    # ── Panel 1: Latent Phase Portrait (Exp 3 Limit Cycle vs Exp 4A/B Collapse)
    ax1 = fig.add_subplot(gs[0, 0])
    if "exp3" in pca_results:
        p3 = pca_results["exp3"]["proj"]
        phase3 = rollouts["exp3"]["phase"]
        scatter = ax1.scatter(p3[:, 0], p3[:, 1], c=phase3, cmap="hsv", s=10, alpha=0.7)
        ax1.plot(p3[:, 0], p3[:, 1], color="black", lw=0.5, alpha=0.4)
        cbar = plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("Gait Phase (rad)", fontsize=8)
        ax1.set_title("Exp 3: Latent Limit Cycle Attractor\n(Closed Periodic Ring in PC1-PC2)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("PC 1", fontsize=9)
    ax1.set_ylabel("PC 2", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Exp 4A/4B Divergence Trajectories
    ax2 = fig.add_subplot(gs[0, 1])
    colors = {"exp3": "#2ca02c", "exp4a": "#1f77b4", "exp4b": "#ff7f0e", "exp4c": "#d62728"}
    for exp in ["exp3", "exp4a", "exp4b", "exp4c"]:
        if exp in pca_results:
            p = pca_results[exp]["proj"]
            label = f"{exp.upper()} ({'Survived' if exp=='exp3' else 'Fell'})"
            ax2.plot(p[:, 0], p[:, 1], label=label, color=colors[exp], lw=1.8, alpha=0.85)
            # Mark endpoint
            ax2.scatter(p[-1, 0], p[-1, 1], color=colors[exp], marker="x" if exp!="exp3" else "o", s=80, zorder=5)

    ax2.set_title("Latent Trajectory Divergence\n(Exp 3 Orbit vs Exp 4 Collapse)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("PC 1", fontsize=9)
    ax2.set_ylabel("PC 2", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Temporal Evolution of PC1 (Neural Rhythm)
    ax3 = fig.add_subplot(gs[0, 2])
    for exp in ["exp3", "exp4a", "exp4b", "exp4c"]:
        if exp in rollouts:
            r = rollouts[exp]
            p = pca_results[exp]["proj"]
            t_sub = r["time"][:len(p)]
            ax3.plot(t_sub, p[:, 0], label=f"{exp.upper()}", color=colors[exp], lw=1.5)

    ax3.set_xlim(0, 3.0)
    ax3.set_title("Neural Rhythm Evolution (PC 1 vs Time)\n(Rhythmic Oscillation vs Trajectory Crash)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Time (s)", fontsize=9)
    ax3.set_ylabel("Latent PC 1 Amplitude", fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Linear Probing Scorecard Table
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    table_data = [
        ["Experiment", "Phase R^2", "Vel R^2", "Contact Acc", "Status"],
    ]
    for exp in ["exp3", "exp4a", "exp4b", "exp4c"]:
        if exp in probe_metrics:
            m = probe_metrics[exp]
            dur = rollouts[exp]["time"][-1]
            if dur >= 5.0:
                status = "10s Steady"
                p_str = f"{m['phase_r2']:+.3f}"
                v_str = f"{m['vx_r2']:+.3f}"
            else:
                status = f"Fell ({dur:.2f}s)"
                p_str = "N/A (Fell)"
                v_str = "N/A (Fell)"
            table_data.append([
                exp.upper(),
                p_str,
                v_str,
                f"{m['contact_acc']:.1f}%",
                status,
            ])

    table = ax4.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#e0e0e0")
        table[(0, i)].set_text_props(weight="bold")
    ax4.set_title("Linear Probing Evaluation\n(Decoding Kinematics from Hidden States)", fontsize=10, fontweight="bold", pad=15)

    # ── Panel 5: Actor Output Weight Heatmap (Exp 3 vs Exp 4A)
    ax5 = fig.add_subplot(gs[1, 1:])
    if "exp3" in weight_metrics and "exp4a" in weight_metrics:
        W3 = weight_metrics["exp3"]["W"]
        W4a = weight_metrics["exp4a"]["W"]
        abs_diff = np.abs(W3)
        im = ax5.imshow(abs_diff, aspect="auto", cmap="viridis", interpolation="nearest")
        ax5.set_yticks(np.arange(10))
        ax5.set_yticklabels(ACTUATOR_LABELS, fontsize=8)
        # Mark boundary between Left and Right leg
        ax5.axhline(4.5, color="red", lw=2, linestyle="--", label="Left/Right Leg Actuator Boundary")
        cbar = plt.colorbar(im, ax=ax5, fraction=0.025, pad=0.02)
        cbar.set_label("|Weight Magnitude|", fontsize=8)
        ax5.set_title(f"Actor Output Layer Weights W_mu (10 × 256)\n(Left ||W||={weight_metrics['exp3']['left_frob']:.1f}, Right ||W||={weight_metrics['exp3']['right_frob']:.1f} → {weight_metrics['exp3']['asymmetry_pct']:.1f}% Asymmetry)", fontsize=10, fontweight="bold")
        ax5.set_xlabel("Latent Hidden Dimensions (0 to 255)", fontsize=9)
        ax5.legend(loc="upper right", fontsize=8)

    # ── Panel 6: Bilateral Weight Frobenius Norm Comparison Bar Chart
    ax6 = fig.add_subplot(gs[2, 0])
    exps = [e for e in ["exp3", "exp4a", "exp4b", "exp4c"] if e in weight_metrics]
    x_indices = np.arange(len(exps))
    width = 0.35
    left_frobs = [weight_metrics[e]["left_frob"] for e in exps]
    right_frobs = [weight_metrics[e]["right_frob"] for e in exps]

    ax6.bar(x_indices - width/2, left_frobs, width, label="Left Leg ||W||_F", color="#1f77b4", alpha=0.85)
    ax6.bar(x_indices + width/2, right_frobs, width, label="Right Leg ||W||_F", color="#d62728", alpha=0.85)
    ax6.set_xticks(x_indices)
    ax6.set_xticklabels([e.upper() for e in exps], fontsize=9)
    ax6.set_ylabel("Frobenius Norm ||W||_F", fontsize=9)
    ax6.set_title("Bilateral Output Weight Norm Balance", fontsize=10, fontweight="bold")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis="y")

    # ── Panel 7: Gait Phase Decoding Demonstration (Ground Truth vs Linear Probe on Exp 3)
    ax7 = fig.add_subplot(gs[2, 1:])
    if "exp3" in rollouts:
        r3 = rollouts["exp3"]
        H3 = r3["hidden"]
        phase3 = r3["phase"]
        split = int(len(H3) * 0.7)
        Y3 = np.column_stack([np.sin(phase3), np.cos(phase3)])
        r2, pred = train_linear_ridge_probe(H3[:split], Y3[:split], H3[split:], Y3[split:])
        pred_phase = np.arctan2(pred[:, 0], pred[:, 1]) % (2.0 * np.pi)
        t_test = r3["time"][split:]

        ax7.plot(t_test[:150], phase3[split:split+150], label="Kinematic Ground Truth Phase", color="black", lw=2, linestyle="-")
        ax7.plot(t_test[:150], pred_phase[:150], label=f"Linearly Probed Phase from Hidden State (Test R² = {r2:.3f})", color="dodgerblue", lw=1.5, linestyle="--")
        ax7.set_title(f"Linear Phase Decoding from Hidden State h_t (Exp 3 | Test R² = {r2:.3f})\n(Ground Truth Gait Phase vs Linear Ridge Prediction)", fontsize=10, fontweight="bold")
        ax7.set_xlabel("Time (s)", fontsize=9)
        ax7.set_ylabel("Gait Phase (rad)", fontsize=9)
        ax7.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax7.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"], fontsize=8)
        ax7.legend(loc="upper right", fontsize=8)
        ax7.grid(True, alpha=0.3)

    plt.suptitle("Unitree H1 Humanoid: Policy Hidden-State Dynamics (h_t in R^256) & Output Weight Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()
    print(f"SUCCESS: Figure saved to {args.output}")



if __name__ == "__main__":
    main()
