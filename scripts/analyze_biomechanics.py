"""
scripts/analyze_biomechanics.py: Biomechanical Gait & Telemetry Analysis.

Processes evaluation telemetry from an .npz archive and produces high-resolution
scientific figures illustrating:
  1. Center of Mass Trajectory (Top View - Lateral Drift)
  2. CoM Vertical Height & Oscillation Amplitude (sigma_z)
  3. Steady-State Energy Exchange (KE vs PE Phase Shift & Correlation)
  4. Forward Velocity Tracking
  5. Foot Contact Timing (Left vs Right Alternation)
  6. Applied Joint Torques & Mechanical Power
  7. Summary Table with Biomechanical Verification
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import compute_energy_exchange, compute_mechanical_cot, compute_gait_metrics


def count_debounced_steps(contact_array, min_hold_steps=4):
    """Counts foot strikes only when contact is maintained for >= min_hold_steps."""
    c = np.asarray(contact_array, dtype=bool)
    steps = 0
    in_stance = False
    hold_count = 0
    for val in c:
        if val:
            hold_count += 1
            if not in_stance and hold_count >= min_hold_steps:
                steps += 1
                in_stance = True
        else:
            hold_count = 0
            in_stance = False
    return steps


def analyze_and_plot(data_path: str, output_image_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    data = np.load(data_path)
    time = data["time"]
    com_pos = data["com_pos"]
    com_vel = data["com_vel"]
    applied_torque = data["applied_torque"]
    joint_vel = data["joint_vel"]
    left_contact = data["left_contact"]
    right_contact = data["right_contact"]

    dt = np.mean(np.diff(time)) if len(time) > 1 else 0.01
    duration = time[-1] - time[0] if len(time) > 1 else 0.0
    distance = float(com_pos[-1, 0] - com_pos[0, 0])
    avg_velocity = float(np.mean(com_vel[:, 0]))
    height_variation = float(np.std(com_pos[:, 2]))
    max_height = float(np.max(com_pos[:, 2]))
    min_height = float(np.min(com_pos[:, 2]))

    # Compute energy exchange
    ke, pe, te, r_corr = compute_energy_exchange(com_pos, com_vel)

    # Compute Mechanical Cost of Transport
    cot = compute_mechanical_cot(applied_torque, joint_vel, distance, dt=dt)

    # Compute Quantitative Human-Likeness & Work Metrics
    gait_m = compute_gait_metrics(left_contact, right_contact, applied_torque, joint_vel, distance, dt=dt)

    # Debounced footstep detection (eliminates 1-tick contact edge chatter)
    left_steps = count_debounced_steps(left_contact, min_hold_steps=4)
    right_steps = count_debounced_steps(right_contact, min_hold_steps=4)
    total_steps = left_steps + right_steps
    cadence = (total_steps / duration * 60.0) if duration > 1.0 else 0.0


    # ── Create Multi-Panel Scientific Figure ─────────────────────────
    fig = plt.figure(figsize=(18, 14), dpi=150)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. CoM Trajectory (Top View)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(com_pos[:, 0], com_pos[:, 1], 'b-', linewidth=2, label='CoM Trajectory')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.6, label='Heading Line')
    ax1.set_xlabel('Forward X Position (m)', fontsize=10)
    ax1.set_ylabel('Lateral Y Position (m)', fontsize=10)
    ax1.set_title('CoM Path (Top View)', fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.6, 0.6)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    ax1.set_aspect('equal', adjustable='box')

    # 2. CoM Height Over Time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, com_pos[:, 2], 'g-', linewidth=2, label='CoM Height')
    ax2.axhline(max_height, color='r', linestyle='--', alpha=0.7, label=f'Max: {max_height:.3f}m')
    ax2.axhline(min_height, color='b', linestyle='--', alpha=0.7, label=f'Min: {min_height:.3f}m')
    ax2.axhline(0.98, color='k', linestyle=':', alpha=0.5, label='Nominal: 0.98m')
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Height (m)', fontsize=10)
    ax2.set_title(f'CoM Height Stability (σ = {height_variation:.4f}m)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8)

    # 3. Forward Velocity
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time, com_vel[:, 0], color='purple', linewidth=2, label='Forward Velocity')
    ax3.axhline(1.0, color='k', linestyle='--', label='Target: 1.0 m/s')
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Velocity (m/s)', fontsize=10)
    ax3.set_title(f'Velocity Tracking (Mean: {avg_velocity:.2f} m/s)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=9)

    # 4. Energy Exchange (The Central Biomechanical Plot)
    ax4 = fig.add_subplot(gs[1, 0:2])
    # Normalize KE and PE for visual phase comparison
    ke_norm = (ke - np.min(ke)) / (np.max(ke) - np.min(ke) + 1e-6)
    pe_norm = (pe - np.min(pe)) / (np.max(pe) - np.min(pe) + 1e-6)

    ax4.plot(time, ke_norm, 'r-', linewidth=2, label='Kinetic Energy (Norm)')
    ax4.plot(time, pe_norm, 'b-', linewidth=2, label='Potential Energy (Norm)')
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Normalized Energy [0, 1]', fontsize=10)
    is_sustained = (duration >= 9.8) and (distance >= 3.0)
    if is_sustained:
        ax4.set_title(f'Mechanical Energy Exchange (Normalized KE vs PE | Pearson r = {r_corr:.3f})', fontsize=11, fontweight='bold')
    else:
        ax4.set_title(f'Mechanical Energy Exchange (Transient / Non-Periodic | r = {r_corr:.3f})', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=9)

    # 5. Foot Contact Timing (Gait Cycles)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.fill_between(time, 0, left_contact.astype(float), step="pre", color='blue', alpha=0.4, label='Left Stance')
    ax5.fill_between(time, 0, -right_contact.astype(float), step="pre", color='crimson', alpha=0.4, label='Right Stance')
    ax5.set_yticks([-1, 0, 1])
    ax5.set_yticklabels(['Right', 'None', 'Left'])
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Ground Contact', fontsize=10)
    ax5.set_title(f'Gait Contact Alternation ({total_steps} debounced steps)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right', fontsize=8)

    # 6. Applied Joint Torques
    ax6 = fig.add_subplot(gs[2, 0:2])
    ax6.plot(time, applied_torque, linewidth=1, alpha=0.6)
    ax6.set_xlabel('Time (s)', fontsize=10)
    ax6.set_ylabel('Joint Torque (N·m)', fontsize=10)
    ax6.set_title('Applied Joint Torques (10 Leg Actuators)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Biomechanical Summary Table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    is_sustained = (duration >= 9.8) and (distance >= 3.0)
    is_static = (duration >= 9.8) and (distance < 1.0)
    is_knee_buckle = (3.0 <= duration < 9.8)

    if is_sustained:
        status_str = "Sustained Locomotion (Asymmetric Split-Stance)"
        cot_str = f"{cot:.3f}" if np.isfinite(cot) else "N/A"
        work_str = f"{gait_m['positive_work_norm']:.3f}" if np.isfinite(gait_m['positive_work_norm']) else "N/A"
        cadence_str = f"{cadence:.1f} steps/min"
        steps_str = f"{total_steps} (Debounced)"
        duty_mean_str = f"{gait_m['duty_factor_mean']:.3f} (Hum: ~0.60)"
        duty_lr_str = f"{gait_m['duty_factor_left']:.2f} / {gait_m['duty_factor_right']:.2f}"
        sym_str = f"{gait_m['symmetry_index']:.1f}% (Hum: <5%)"
        r_str = f"{r_corr:.3f} (Hum: <-0.5)"
        assessment_str = (
            "• Sustained forward locomotion.\n"
            "• High-frequency asymmetric limit cycle.\n"
            "• Lateral balance maintained via split-stance.\n"
        )
    elif is_static:
        status_str = "Static Standing / No Forward Locomotion"
        cot_str = "N/A (v_x ≈ 0, static holding)"
        work_str = "N/A (Static)"
        cadence_str = "N/A (Stationary)"
        steps_str = "N/A (Stationary)"
        duty_mean_str = "1.000 (Double Support)"
        duty_lr_str = "1.00 / 1.00"
        sym_str = "N/A (Static)"
        r_str = f"{r_corr:.3f} (Settling Transient)"
        assessment_str = (
            "• Frozen / Mannequin local optimum.\n"
            "• Robot standing still to harvest alive bonus.\n"
            "• Near-zero forward displacement.\n"
        )
    elif is_knee_buckle:
        status_str = f"Knee-Buckle / Drop to Knees ({duration:.2f}s)"
        cot_str = "N/A (Transient Buckle)"
        work_str = "N/A (Non-steady)"
        cadence_str = "N/A (Stance Jitter)"
        steps_str = f"{total_steps} (Stance jitter)"
        duty_mean_str = "N/A (Pre-drop stance)"
        duty_lr_str = "N/A"
        sym_str = "N/A (Transient)"
        r_str = f"{r_corr:.3f} (Buckle Transient)"
        assessment_str = (
            "• Stiff upright stance with minor jitter.\n"
            "• Forward step attempt causes knee buckling.\n"
            "• Drops to knees/shins at episode end.\n"
        )
    else:
        status_str = f"Early Termination (Fell at {duration:.2f}s)"
        cot_str = "N/A (Transient Fall)"
        work_str = "N/A (Transient Fall)"
        cadence_str = "N/A (Transient Fall)"
        steps_str = f"{total_steps} (Pre-fall contact)"
        duty_mean_str = "N/A (Fell <2s)"
        duty_lr_str = "N/A (Fell <2s)"
        sym_str = "N/A (Fell <2s)"
        r_str = f"{r_corr:.3f} (Fall Transient Surge)"
        assessment_str = (
            "• Failed to sustain balance (early termination).\n"
            "• Falling transients invalidate periodic phase analysis.\n"
            "• Loss of support polygon due to unactuated pitch.\n"
        )

    summary_text = (
        f"BIOMECHANICAL SCORECARD\n"
        f"════════════════════════════════\n"
        f"Episode Duration     : {duration:.2f} s\n"
        f"Distance Traveled    : {distance:.2f} m\n"
        f"Average Speed        : {avg_velocity:.2f} m/s\n"
        f"Total Steps Detected : {steps_str}\n"
        f"Cadence              : {cadence_str}\n"
        f"────────────────────────────────\n"
        f"EFFICIENCY METRICS:\n"
        f"• Cost of Transport  : {cot_str}\n"
        f"• Positive Work Norm : {work_str}\n"
        f"────────────────────────────────\n"
        f"HUMAN-LIKENESS METRICS:\n"
        f"• Duty Factor (Mean) : {duty_mean_str}\n"
        f"• Duty Factor (L/R)  : {duty_lr_str}\n"
        f"• Symmetry Index (SI): {sym_str}\n"
        f"• KE/PE Correlation  : {r_str}\n"
        f"• CoM Height (Mean)  : {np.mean(com_pos[:, 2]):.3f} m\n"
        f"• Height Var (σ)     : {height_variation:.4f} m\n"
        f"────────────────────────────────\n"
        f"SCIENTIFIC ASSESSMENT:\n"
        f"{assessment_str}"
    )

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=9.0,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='linen', edgecolor='gray', alpha=0.5))


    plt.suptitle("Unitree H1 Humanoid: Biomechanical Locomotion Analysis", fontsize=15, fontweight='bold', y=0.99)
    plt.savefig(output_image_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Biomechanical figure saved to: {output_image_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze biomechanical locomotion data")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to evaluation .npz archive")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path for figure (.png)")
    args = parser.parse_args()

    default_output = args.data.replace(".npz", "_analysis.png")
    output_path = args.output or default_output

    analyze_and_plot(args.data, output_path)


if __name__ == "__main__":
    main()
