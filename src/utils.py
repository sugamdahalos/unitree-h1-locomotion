"""
Scientific utility functions for kinematics, coordinate transformations,
gait cycle segmentation, and biomechanical metric calculations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def quat_to_rot_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)
    return R


def project_gravity(q: np.ndarray) -> np.ndarray:
    """
    Projects world gravity unit vector [0, 0, -1] into the robot base body frame.
    Returns:
        3D vector in body frame representing projected gravity.
        When robot is perfectly upright, returns [0, 0, -1].
    """
    R = quat_to_rot_mat(q)
    # R transforms from body to world; R.T transforms from world to body
    world_gravity = np.array([0.0, 0.0, -1.0])
    return R.T @ world_gravity


def compute_mechanical_cot(
    torques: np.ndarray,
    joint_velocities: np.ndarray,
    distance_traveled: float,
    mass: float = 47.0,
    gravity: float = 9.81,
    dt: float = 0.01,
) -> float:
    """
    Calculate dimensionless Mechanical Cost of Transport (CoT).

        CoT = Total Mechanical Work / (Mass * Gravity * Distance)
        Total Work = integral of sum(|tau_i * qvel_i|) dt

    Args:
        torques: Array of shape (N, num_actuators) of applied joint torques (Nm).
        joint_velocities: Array of shape (N, num_actuators) of joint angular velocities (rad/s).
        distance_traveled: Net forward displacement (m).
        mass: Mass of Unitree H1 in kg (~47 kg).
        gravity: Gravitational acceleration (9.81 m/s^2).
        dt: Control timestep in seconds.

    Returns:
        Scalar Cost of Transport value (dimensionless).
    """
    if distance_traveled <= 0.05:
        return float("inf")

    # Mechanical power = sum(|tau * omega|) across all actuators
    mechanical_power = np.sum(np.abs(torques * joint_velocities), axis=-1)
    total_work = np.sum(mechanical_power) * dt
    cot = total_work / (mass * gravity * distance_traveled)
    return float(cot)


def compute_energy_exchange(
    com_pos: np.ndarray,
    com_vel: np.ndarray,
    mass: float = 47.0,
    gravity: float = 9.81,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute translational Kinetic Energy, Potential Energy, Total Energy,
    and the KE/PE Pearson phase correlation.

    In biological inverted-pendulum walking, Kinetic and Potential energy exchange
    nearly out of phase (correlation typically -0.5 to -0.85).

    Returns:
        ke: Kinetic energy series (Joules)
        pe: Potential energy series (Joules)
        te: Total mechanical energy series (Joules)
        correlation: Pearson correlation coefficient between KE and PE
    """
    ke = 0.5 * mass * np.sum(com_vel**2, axis=-1)
    pe = mass * gravity * com_pos[:, 2]
    te = ke + pe

    std_ke = np.std(ke)
    std_pe = np.std(pe)

    if std_ke > 1e-4 and std_pe > 1e-4:
        ke_norm = (ke - np.mean(ke)) / std_ke
        pe_norm = (pe - np.mean(pe)) / std_pe
        correlation = float(np.corrcoef(ke_norm, pe_norm)[0, 1])
    else:
        correlation = 0.0

    return ke, pe, te, correlation


def segment_gait_cycles(
    contact_left: np.ndarray,
    contact_right: np.ndarray,
    dt: float = 0.01,
) -> List[Tuple[int, int]]:
    """
    Segment continuous locomotion into individual stride cycles using foot contact transitions.
    A full stride is defined from consecutive left heel-strike / ground-touchdown events.

    Returns:
        List of (start_step, end_step) index pairs for each valid gait cycle.
    """
    # Detect transitions from not contacting (0) to contacting (1)
    touchdowns = np.where((contact_left[:-1] == 0) & (contact_left[1:] == 1))[0] + 1
    strides = []
    for i in range(len(touchdowns) - 1):
        start = touchdowns[i]
        end = touchdowns[i + 1]
        stride_duration = (end - start) * dt
        # Plausible human/humanoid stride duration is 0.4s to 2.0s
        if 0.35 <= stride_duration <= 2.0:
            strides.append((start, end))
    return strides


def compute_gait_metrics(
    contact_left: np.ndarray,
    contact_right: np.ndarray,
    applied_torques: np.ndarray,
    joint_velocities: np.ndarray,
    distance_traveled: float,
    dt: float = 0.01,
    mass: float = 47.0,
    gravity: float = 9.81,
) -> Dict[str, float]:
    """
    Compute quantitative human-likeness and mechanical work metrics:
      - duty_factor_left, duty_factor_right: Stance phase fraction (Human normal: ~0.60-0.62)
      - symmetry_index: Left/Right stance asymmetry percentage (Human normal: < 5%)
      - positive_work_norm: Dimensionless positive mechanical work = W^+ / (m * g * d)
    """
    total_steps = len(contact_left)
    if total_steps == 0:
        return {
            "duty_factor_left": 0.0,
            "duty_factor_right": 0.0,
            "duty_factor_mean": 0.0,
            "symmetry_index": 0.0,
            "positive_work_norm": float("inf"),
        }

    left_stance_time = float(np.sum(contact_left)) * dt
    right_stance_time = float(np.sum(contact_right)) * dt
    total_time = total_steps * dt

    df_left = left_stance_time / total_time if total_time > 0 else 0.0
    df_right = right_stance_time / total_time if total_time > 0 else 0.0
    df_mean = 0.5 * (df_left + df_right)

    # Gait Symmetry Index (SI)
    if (df_left + df_right) > 1e-4:
        si = (abs(df_left - df_right) / (0.5 * (df_left + df_right))) * 100.0
    else:
        si = 0.0

    # Positive mechanical work: integral of sum(max(0, tau * qvel)) dt
    power = applied_torques * joint_velocities
    positive_power = np.maximum(0.0, power)
    total_pos_work = float(np.sum(positive_power) * dt)

    if distance_traveled > 0.05:
        pos_work_norm = total_pos_work / (mass * gravity * distance_traveled)
    else:
        pos_work_norm = float("inf")

    return {
        "duty_factor_left": df_left,
        "duty_factor_right": df_right,
        "duty_factor_mean": df_mean,
        "symmetry_index": si,
        "positive_work_norm": pos_work_norm,
    }
