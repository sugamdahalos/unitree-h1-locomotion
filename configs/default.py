"""
Centralized Configuration for Unitree H1 Locomotion Research Framework.

Defines reproducible hyperparameters, physics parameters, and reward specifications
for the 6 research ablation experiments:
  - Exp 1 (exp1_naive): Naive velocity tracking baseline (stiff stance & knee-buckle collapse).
  - Exp 2 (exp2_posture): Posture & effort regularization with alive bonus (frozen mannequin trap).
  - Exp 3 (exp3_effort): Progress + effort minimization with zero alive bonus (asymmetric scooter exploit).
  - Exp 4A (exp4a_symmetry): Exp 3 + bilateral effort/action symmetry penalty.
  - Exp 4B (exp4b_clearance): Exp 3 + swing-foot ground clearance penalty (anti-dragging).
  - Exp 4C (exp4c_alternation): Exp 3 + alternating left/right contact reward.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

# ── Base Directory Setup ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_XML_PATH = os.path.join(PROJECT_ROOT, "models", "unitree_h1", "scene.xml")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
TB_LOG_DIR = os.path.join(PROJECT_ROOT, "tb_logs")


@dataclass
class RewardWeights:
    """
    Modular reward weights representing the biomechanical and task objectives.
    All terms are defined such that positive weights scale rewards, and penalties are subtracted.
    """
    # Task objectives
    velocity_weight: float = 1.0       # Weight for forward velocity tracking
    velocity_tracking_mode: str = "exponential"  # 'exponential' (Exp 1/2) or 'linear' (Exp 3/4/5)
    survival_bonus: float = 0.5        # Instantaneous bonus for remaining upright
    fall_penalty: float = 10.0         # Penalty incurred upon early termination (fall)

    # Base posture & stability
    orientation_weight: float = 0.0    # Penalty on torso tilt away from gravity vector (pitch/roll)
    target_height_weight: float = 0.0  # Penalty on (z - z_target)^2 for level CoM vaulting
    vertical_vel_weight: float = 0.0   # Penalty on vertical velocity qvel_z^2 (anti-jumping)

    # Actuation & smoothness regularization
    energy_weight: float = 0.001       # Penalty on sum(tau^2) or sum(action^2)
    smoothness_weight: float = 0.0     # Penalty on sum((action_t - action_{t-1})^2)

    # Stepping & contact dynamics
    air_time_weight: float = 0.0       # Bonus for achieving nominal single-stance swing duration
    foot_clearance_weight: float = 0.0 # Penalty for swing foot failing to clear floor (Exp 4B)
    symmetry_weight: float = 0.0       # Penalty on bilateral leg effort/action asymmetry (Exp 4A)
    contact_alternation_weight: float = 0.0  # Reward for alternating left/right touchdown steps (Exp 4C)


@dataclass
class ExperimentConfig:
    """Complete experimental configuration schema."""
    name: str = "exp1_naive"
    description: str = "Naive forward velocity tracking baseline"

    # Physics & Simulation
    xml_path: str = MODEL_XML_PATH
    frame_skip: int = 5                 # 5 * 0.002s = 0.01s (100 Hz control frequency)
    max_episode_steps: int = 1000       # 10.0 seconds of simulation
    target_velocity: float = 1.0        # Desired forward speed (m/s)
    nominal_com_height: float = 0.98    # Nominal standing CoM height in meters
    healthy_z_min: float = 0.60         # Termination threshold (fall detected)
    healthy_z_max: float = 1.50         # Termination threshold (excessive launch)

    # Actuation Mode
    legs_only: bool = True              # If True, policy controls 10 leg actuators; upper body has PD holding
    upper_body_kp: float = 100.0        # Proportional gain for upper-body posture holding
    upper_body_kd: float = 5.0          # Derivative gain for upper-body damping

    # Rewards
    rewards: RewardWeights = field(default_factory=RewardWeights)

    # SAC Training Hyperparameters
    seed: int = 42
    n_envs: int = 12                    # Number of parallel environments (SubprocVecEnv, 12 cores)
    total_timesteps: int = 3_000_000
    learning_starts: int = 50_000
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    gradient_steps: int = 8             # Update gradient steps per vectorized step
    train_freq: int = 1
    net_arch: List[int] = field(default_factory=lambda: [256, 256])

    # Logging & Checkpoints
    eval_freq: int = 25_000             # Steps between evaluation rounds
    n_eval_episodes: int = 10           # Number of episodes per evaluation
    save_freq: int = 100_000            # Periodic checkpoint save frequency
    tb_log_dir: str = TB_LOG_DIR
    checkpoint_dir: str = CHECKPOINT_DIR


# ── Canonical Experiment Definitions ──────────────────────────────

def get_exp1_config() -> ExperimentConfig:
    """
    Experiment 1: Naive Baseline
    Exponential velocity tracking + alive bonus with minimal torque penalty.
    Result: Robot stands stiffly with minor jitter before knees buckle at ~5-9s.
    """
    cfg = ExperimentConfig(
        name="exp1_naive",
        description="Experiment 1: Naive velocity tracking baseline (stiff stance & knee-buckle)",
    )
    cfg.rewards = RewardWeights(
        velocity_weight=1.0,
        survival_bonus=0.5,
        fall_penalty=10.0,
        energy_weight=0.001,
        orientation_weight=0.0,
        target_height_weight=0.0,
        vertical_vel_weight=0.0,
        smoothness_weight=0.0,
        air_time_weight=0.0,
        foot_clearance_weight=0.0,
    )
    return cfg


def get_exp2_config() -> ExperimentConfig:
    """
    Experiment 2: Posture & Regularization Priors (+ Alive Bonus)
    Adds torso orientation alignment, CoM height penalty, vertical velocity penalty,
    and action rate smoothness while retaining the +0.5/step survival bonus.
    Result: Converges to the 'Frozen Mannequin' local optimum (stands still to farm survival bonus).
    """
    cfg = ExperimentConfig(
        name="exp2_posture",
        description="Experiment 2: Posture & effort regularization with alive bonus (frozen mannequin trap)",
    )
    cfg.rewards = RewardWeights(
        velocity_weight=1.0,
        survival_bonus=0.5,
        fall_penalty=10.0,
        energy_weight=0.002,
        orientation_weight=1.5,
        target_height_weight=5.0,
        vertical_vel_weight=2.0,
        smoothness_weight=0.1,
        air_time_weight=0.0,
        foot_clearance_weight=0.0,
    )
    return cfg


def get_exp3_config() -> ExperimentConfig:
    """
    Experiment 3: Decoupling Test (Task Progress + Effort Cost, ZERO Alive Bonus)
    Tests whether removing the alive bonus forces movement and whether torque/smoothness
    penalties eliminate explosive hopping without any postural priors.
    """
    cfg = ExperimentConfig(
        name="exp3_effort",
        description="Experiment 3: Linear progress + effort minimization (zero alive bonus)",
    )
    cfg.rewards = RewardWeights(
        velocity_weight=2.0,
        velocity_tracking_mode="linear",
        survival_bonus=0.0,           # ZERO ALIVE BONUS - standing still earns 0
        fall_penalty=10.0,
        energy_weight=0.002,          # Torque minimization
        smoothness_weight=0.02,       # Action rate regularization
        orientation_weight=0.0,       # Unconstrained posture
        target_height_weight=0.0,     # Unconstrained height
        vertical_vel_weight=0.0,      # Unconstrained vertical velocity
        air_time_weight=0.0,
        foot_clearance_weight=0.0,
    )
    return cfg


def get_exp4a_config() -> ExperimentConfig:
    """
    Experiment 4A: Symmetry Only (Exp 3 + Weak Bilateral Symmetry Objective)
    Tests whether penalizing left/right actuation asymmetry alone eliminates
    the scooter gait without imposing any foot clearance or timing constraints.
    """
    cfg = ExperimentConfig(
        name="exp4a_symmetry",
        description="Experiment 4A: Exp 3 + bilateral effort/action symmetry penalty",
    )
    cfg.rewards = RewardWeights(
        velocity_weight=2.0,
        velocity_tracking_mode="linear",
        survival_bonus=0.0,           # ZERO ALIVE BONUS
        fall_penalty=10.0,
        energy_weight=0.002,          # Torque minimization
        smoothness_weight=0.02,       # Action rate regularization
        orientation_weight=0.0,       # Unconstrained posture
        target_height_weight=0.0,     # Unconstrained height
        vertical_vel_weight=0.0,      # Unconstrained vertical velocity
        air_time_weight=0.0,          # No air time prior
        foot_clearance_weight=0.0,    # No clearance penalty
        symmetry_weight=0.01,         # WEAK BILATERAL SYMMETRY PENALTY ONLY
        contact_alternation_weight=0.0,
    )
    return cfg


def get_exp4b_config() -> ExperimentConfig:
    """
    Experiment 4B: Swing Clearance Only (Exp 3 + Swing Foot Ground Clearance)
    Tests whether simply penalizing the swing foot for failing to clear the ground
    (anti-dragging) forces the robot out of the scooter shuffle without symmetry or timing priors.
    """
    cfg = ExperimentConfig(
        name="exp4b_clearance",
        description="Experiment 4B: Exp 3 + swing-foot ground clearance penalty (anti-dragging)",
    )
    cfg.rewards = RewardWeights(
        velocity_weight=2.0,
        velocity_tracking_mode="linear",
        survival_bonus=0.0,           # ZERO ALIVE BONUS
        fall_penalty=10.0,
        energy_weight=0.002,          # Torque minimization
        smoothness_weight=0.02,       # Action rate regularization
        orientation_weight=0.0,       # Unconstrained posture
        target_height_weight=0.0,     # Unconstrained height
        vertical_vel_weight=0.0,      # Unconstrained vertical velocity
        air_time_weight=0.0,          # No air time prior
        foot_clearance_weight=0.3,    # MINIMAL SWING FOOT CLEARANCE PENALTY ONLY
        symmetry_weight=0.0,          # No symmetry prior
        contact_alternation_weight=0.0,
    )
    return cfg


def get_exp4c_config() -> ExperimentConfig:
    """
    Experiment 4C: Contact Alternation Only (Exp 3 + Alternating Left/Right Support)
    Tests whether rewarding sequential alternating touchdowns (L -> R -> L -> R)
    alone forces natural periodic locomotion without joint trajectories or clearance priors.
    """
    cfg = ExperimentConfig(
        name="exp4c_alternation",
        description="Experiment 4C: Exp 3 + alternating left/right contact reward",
    )
    cfg.rewards = RewardWeights(
        velocity_weight=2.0,
        velocity_tracking_mode="linear",
        survival_bonus=0.0,           # ZERO ALIVE BONUS
        fall_penalty=10.0,
        energy_weight=0.002,          # Torque minimization
        smoothness_weight=0.02,       # Action rate regularization
        orientation_weight=0.0,       # Unconstrained posture
        target_height_weight=0.0,     # Unconstrained height
        vertical_vel_weight=0.0,      # Unconstrained vertical velocity
        air_time_weight=0.0,          # No air time prior
        foot_clearance_weight=0.0,    # No clearance penalty
        symmetry_weight=0.0,          # No symmetry prior
        contact_alternation_weight=0.5, # CONTACT ALTERNATION REWARD ONLY
    )
    return cfg


def get_exp5_config() -> ExperimentConfig:
    """
    Experiment 5: Biological Prior Combination (Symmetry + Clearance + Alternation)
    Tests whether combining the minimal physical and temporal priors produces human-like
    duty factor (~0.60), low cadence (~100-120 steps/min), and bilateral symmetry (<5%).
    """
    cfg = ExperimentConfig(
        name="exp5_stepping",
        description="Experiment 5: Combined minimal priors (clearance + alternation + symmetry)",
    )
    cfg.rewards = RewardWeights(
        velocity_weight=2.0,
        velocity_tracking_mode="linear",
        survival_bonus=0.0,           # ZERO ALIVE BONUS
        fall_penalty=10.0,
        energy_weight=0.002,
        smoothness_weight=0.02,
        orientation_weight=0.5,
        target_height_weight=0.5,
        vertical_vel_weight=0.2,
        foot_clearance_weight=0.3,
        symmetry_weight=0.01,
        contact_alternation_weight=0.5,
        air_time_weight=0.5,
    )
    return cfg


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "exp1": get_exp1_config(),
    "exp2": get_exp2_config(),
    "exp3": get_exp3_config(),
    "exp4": get_exp4a_config(),
    "exp4a": get_exp4a_config(),
    "exp4a_symmetry": get_exp4a_config(),
    "exp4b": get_exp4b_config(),
    "exp4b_clearance": get_exp4b_config(),
    "exp4c": get_exp4c_config(),
    "exp4c_alternation": get_exp4c_config(),
    "exp5": get_exp5_config(),
}


def get_config(name: str = "exp1") -> ExperimentConfig:
    """Lookup experiment configuration by name or alias."""
    key = name.lower().strip()
    if key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{name}'. Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[key]

