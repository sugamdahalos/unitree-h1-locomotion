"""
Gymnasium environment for Unitree H1 bipedal locomotion in MuJoCo.

Features:
  - Active Upper-Body PD Holding: Prevents the unactuated ragdoll defect when training legs-only.
  - Markovian Observation Space: Projected gravity, base velocities, joint kinematics, and action memory.
  - Modular Biomechanical Rewards: Configurable via ExperimentConfig (Exp 1 to Exp 4C).
  - Ground Truth Foot Contact Dynamics: Contact sensor tracking for left/right gait cycle analysis.
"""

from typing import Any, Dict, Optional, Tuple
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from configs.default import ExperimentConfig, get_config
from src.utils import project_gravity, quat_to_rot_mat


# Actuator names for the 10 leg degrees of freedom
LEG_ACTUATORS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]

# Actuator names for the 9 upper-body degrees of freedom
UPPER_ACTUATORS = [
    "torso",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
]


class UnitreeH1Env(gym.Env):
    """
    Gymnasium environment for Unitree H1 humanoid locomotion in MuJoCo.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = config if config is not None else get_config("exp1")
        self.render_mode = render_mode
        self.target_velocity = self.cfg.target_velocity

        # ── Load MuJoCo Model ─────────────────────────────────────────
        if not os.path.exists(self.cfg.xml_path):
            raise FileNotFoundError(f"MuJoCo model XML not found at {self.cfg.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = self.cfg.frame_skip
        self.dt = self.model.opt.timestep * self.frame_skip  # Control timestep (0.01s)

        # ── Map Actuator Indices & Torque Limits ──────────────────────
        self.leg_actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in LEG_ACTUATORS
        ], dtype=np.int32)

        self.upper_actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in UPPER_ACTUATORS
        ], dtype=np.int32)

        if self.cfg.legs_only:
            self.policy_actuator_ids = self.leg_actuator_ids
        else:
            self.policy_actuator_ids = np.arange(self.model.nu, dtype=np.int32)

        self.nu = len(self.policy_actuator_ids)

        # Torque limits for scaling policy actions
        self.ctrl_min = self.model.actuator_ctrlrange[self.policy_actuator_ids, 0].astype(np.float32)
        self.ctrl_max = self.model.actuator_ctrlrange[self.policy_actuator_ids, 1].astype(np.float32)

        # Upper body torque limits
        self.upper_ctrl_min = self.model.actuator_ctrlrange[self.upper_actuator_ids, 0].astype(np.float32)
        self.upper_ctrl_max = self.model.actuator_ctrlrange[self.upper_actuator_ids, 1].astype(np.float32)

        # Upper body qpos and qvel addresses
        self.upper_qpos_adrs = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[aid, 0]]
            for aid in self.upper_actuator_ids
        ], dtype=np.int32)

        self.upper_qvel_adrs = np.array([
            self.model.jnt_dofadr[self.model.actuator_trnid[aid, 0]]
            for aid in self.upper_actuator_ids
        ], dtype=np.int32)

        # Policy joint addresses
        self.policy_qpos_adrs = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[aid, 0]]
            for aid in self.policy_actuator_ids
        ], dtype=np.int32)

        self.policy_qvel_adrs = np.array([
            self.model.jnt_dofadr[self.model.actuator_trnid[aid, 0]]
            for aid in self.policy_actuator_ids
        ], dtype=np.int32)

        # ── Foot Contact Geoms ────────────────────────────────────────
        self.left_foot_geom_ids = set()
        self.right_foot_geom_ids = set()
        self.left_ankle_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link")
        self.right_ankle_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link")
        floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.floor_geom_id = floor_gid if floor_gid != -1 else 0

        for gid in range(self.model.ngeom):
            bid = self.model.geom_bodyid[gid]
            if bid == self.left_ankle_bid:
                self.left_foot_geom_ids.add(gid)
            elif bid == self.right_ankle_bid:
                self.right_foot_geom_ids.add(gid)

        # ── Define Spaces ─────────────────────────────────────────────
        # Action space: normalized in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)

        # Observation space components:
        # 3 (projected gravity) + 3 (base lin vel) + 3 (base ang vel) + 1 (cmd vel)
        # + nu (joint pos) + nu (joint vel) + nu (prev action)
        obs_dim = 3 + 3 + 3 + 1 + self.nu + self.nu + self.nu
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── State Buffers ─────────────────────────────────────────────
        self.step_count = 0
        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self.applied_torque = np.zeros(self.model.nu, dtype=np.float32)
        self.left_air_time = 0.0
        self.right_air_time = 0.0
        self.left_contact_history = False
        self.right_contact_history = False
        self.last_touchdown_foot: Optional[str] = None

        # Visualizer
        self.viewer = None

    def set_target_velocity(self, v_target: float) -> None:
        """Update the target forward walking speed (supports curriculum learning)."""
        self.target_velocity = float(v_target)

    def _get_obs(self) -> np.ndarray:
        """Construct Markovian observation vector localized to base coordinate frame."""
        # 1. Base orientation via projected gravity
        quat = self.data.qpos[3:7]
        proj_gravity = project_gravity(quat).astype(np.float32)

        # 2. Base velocities in local frame (transformed from world to base)
        R = quat_to_rot_mat(quat)
        base_lin_vel = (R.T @ self.data.qvel[0:3]).astype(np.float32)
        base_ang_vel = (R.T @ self.data.qvel[3:6]).astype(np.float32)

        # 3. Target command
        cmd = np.array([self.target_velocity], dtype=np.float32)

        # 4. Joint positions & velocities of actuated joints
        joint_pos = self.data.qpos[self.policy_qpos_adrs].astype(np.float32)
        joint_vel = self.data.qvel[self.policy_qvel_adrs].astype(np.float32)

        obs = np.concatenate([
            proj_gravity,
            base_lin_vel,
            base_ang_vel,
            cmd,
            joint_pos,
            joint_vel,
            self.prev_action,
        ])
        return obs

    def _detect_foot_contacts(self) -> Tuple[bool, bool]:
        """Check MuJoCo contact buffer for contacts between feet and ground plane."""
        left_contact = False
        right_contact = False

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if g1 == self.floor_geom_id or g2 == self.floor_geom_id:
                other_geom = g2 if g1 == self.floor_geom_id else g1
                if other_geom in self.left_foot_geom_ids:
                    left_contact = True
                elif other_geom in self.right_foot_geom_ids:
                    right_contact = True

        return left_contact, right_contact

    def _compute_rewards(
        self,
        action: np.ndarray,
        left_contact: bool,
        right_contact: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute modular reward terms and penalty breakdown based on config.
        """
        rw = self.cfg.rewards

        # 1. Forward Velocity Tracking
        v_forward = float(self.data.qvel[0])
        if getattr(rw, "velocity_tracking_mode", "exponential") == "linear":
            # Linear forward progress: 0 when stationary, negative if moving backward, capped at target speed
            r_vel = rw.velocity_weight * float(np.clip(v_forward, -0.5, self.target_velocity))
        else:
            # Exponential tracking kernel: strictly bounded in [0, 1]
            vel_error = abs(v_forward - self.target_velocity)
            r_vel = rw.velocity_weight * float(np.exp(- (vel_error ** 2) / 0.25))

        # Lateral and yaw suppression (keeps the robot moving straight)
        v_lateral = float(self.data.qvel[1])
        w_yaw = float(self.data.qvel[5])
        r_drift = - 1.0 * (v_lateral ** 2) - 0.5 * (w_yaw ** 2)

        # 2. Survival Bonus
        r_alive = rw.survival_bonus

        # 3. Base Posture & Stability
        # Orientation: penalize tilt away from vertical [0, 0, -1]
        quat = self.data.qpos[3:7]
        proj_g = project_gravity(quat)
        # Tilt error is magnitude of horizontal gravity components
        tilt_error = float(proj_g[0]**2 + proj_g[1]**2)
        r_orientation = - rw.orientation_weight * tilt_error

        # Level CoM height: penalize vertical bouncing to encourage inverted pendulum vaulting
        torso_z = float(self.data.qpos[2])
        height_error = (torso_z - self.cfg.nominal_com_height) ** 2
        r_height = - rw.target_height_weight * height_error

        # Anti-jump vertical velocity penalty
        v_z = float(self.data.qvel[2])
        r_jump = - rw.vertical_vel_weight * (v_z ** 2)

        # 4. Effort & Smoothness Regularization
        torque_norm = float(np.sum((self.applied_torque / np.maximum(self.model.actuator_ctrlrange[:, 1], 1.0)) ** 2))
        r_energy = - rw.energy_weight * torque_norm

        smoothness_error = float(np.sum((action - self.prev_action) ** 2))
        r_smoothness = - rw.smoothness_weight * smoothness_error

        # 5. Stepping Dynamics & Air-Time Prior
        # Reward single-stance air times in the nominal biological window (0.25s to 0.45s)
        r_air_time = 0.0
        if rw.air_time_weight > 0.0:
            # When left foot just made touchdown
            if left_contact and not self.left_contact_history:
                if 0.25 <= self.left_air_time <= 0.45:
                    r_air_time += rw.air_time_weight
            # When right foot just made touchdown
            if right_contact and not self.right_contact_history:
                if 0.25 <= self.right_air_time <= 0.45:
                    r_air_time += rw.air_time_weight

        # Foot clearance bonus: during swing phase, penalize feet dragging on the ground
        r_clearance = 0.0
        if rw.foot_clearance_weight > 0.0:
            left_foot_z = float(self.data.xpos[self.left_ankle_bid, 2])
            right_foot_z = float(self.data.xpos[self.right_ankle_bid, 2])
            if not left_contact and left_foot_z < 0.04:
                r_clearance -= rw.foot_clearance_weight * (0.04 - left_foot_z)
            if not right_contact and right_foot_z < 0.04:
                r_clearance -= rw.foot_clearance_weight * (0.04 - right_foot_z)

        # 6. Bilateral Action/Effort Symmetry Prior (Exp 4A)
        r_sym = 0.0
        if rw.symmetry_weight > 0.0:
            left_act = action[:5]
            right_act = action[5:]
            asym_effort = float((np.sum(left_act ** 2) - np.sum(right_act ** 2)) ** 2)
            r_sym = - rw.symmetry_weight * asym_effort

        # 7. Contact Alternation Prior (Exp 4C)
        r_alt = 0.0
        if rw.contact_alternation_weight > 0.0:
            left_touchdown = left_contact and not self.left_contact_history
            right_touchdown = right_contact and not self.right_contact_history
            if left_touchdown and not right_touchdown:
                if self.last_touchdown_foot == "right":
                    r_alt += rw.contact_alternation_weight
                self.last_touchdown_foot = "left"
            elif right_touchdown and not left_touchdown:
                if self.last_touchdown_foot == "left":
                    r_alt += rw.contact_alternation_weight
                self.last_touchdown_foot = "right"

        # Total step reward
        total_reward = (
            r_vel
            + r_drift
            + r_alive
            + r_orientation
            + r_height
            + r_jump
            + r_energy
            + r_smoothness
            + r_air_time
            + r_clearance
            + r_sym
            + r_alt
        )

        breakdown = {
            "reward_velocity": r_vel,
            "reward_drift": r_drift,
            "reward_alive": r_alive,
            "reward_orientation": r_orientation,
            "reward_height": r_height,
            "reward_jump": r_jump,
            "reward_energy": r_energy,
            "reward_smoothness": r_smoothness,
            "reward_air_time": r_air_time,
            "reward_clearance": r_clearance,
            "reward_symmetry": r_sym,
            "reward_alternation": r_alt,
            "total_reward": total_reward,
        }
        return total_reward, breakdown


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # ── Apply Actuator Controls ───────────────────────────────────
        # Scale policy action [-1, 1] to actuator torque limits
        scaled_policy_ctrl = self.ctrl_min + (action + 1.0) * 0.5 * (self.ctrl_max - self.ctrl_min)

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.policy_actuator_ids] = scaled_policy_ctrl

        # Active Upper-Body PD Holding Controller (when legs_only is True)
        if self.cfg.legs_only:
            upper_qpos = self.data.qpos[self.upper_qpos_adrs]
            upper_qvel = self.data.qvel[self.upper_qvel_adrs]
            # Reference upright posture is 0 rad for all upper joints
            pd_torques = - self.cfg.upper_body_kp * upper_qpos - self.cfg.upper_body_kd * upper_qvel
            pd_torques_clipped = np.clip(pd_torques, self.upper_ctrl_min, self.upper_ctrl_max)
            self.data.ctrl[self.upper_actuator_ids] = pd_torques_clipped

        self.applied_torque = self.data.ctrl.copy()

        # Advance MuJoCo Physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # ── Foot Contact & Air Time Tracking ──────────────────────────
        left_contact, right_contact = self._detect_foot_contacts()

        if left_contact:
            self.left_air_time = 0.0
        else:
            self.left_air_time += self.dt

        if right_contact:
            self.right_air_time = 0.0
        else:
            self.right_air_time += self.dt

        # ── Reward Computation ────────────────────────────────────────
        reward, breakdown = self._compute_rewards(action, left_contact, right_contact)

        # ── Termination Criteria ──────────────────────────────────────
        torso_z = float(self.data.qpos[2])
        proj_gz = project_gravity(self.data.qpos[3:7])[2]

        is_healthy = (self.cfg.healthy_z_min <= torso_z <= self.cfg.healthy_z_max) and (proj_gz < -0.2)
        is_finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()

        terminated = not (is_healthy and is_finite)
        truncated = self.step_count >= self.cfg.max_episode_steps

        # Apply terminal fall penalty upon early failure
        if terminated and not truncated:
            reward -= self.cfg.rewards.fall_penalty
            breakdown["reward_fall_penalty"] = - self.cfg.rewards.fall_penalty

        self.prev_action = action.copy()
        self.left_contact_history = left_contact
        self.right_contact_history = right_contact

        obs = self._get_obs()

        info = {
            "reward_breakdown": breakdown,
            "telemetry": {
                "forward_velocity": float(self.data.qvel[0]),
                "com_height": torso_z,
                "distance_traveled": float(self.data.qpos[0]),
                "step_count": self.step_count,
            },
            "contact": {
                "left": left_contact,
                "right": right_contact,
            }
        }

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Initialize from home keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            self.data.qpos[2] = self.cfg.nominal_com_height
            self.data.qpos[3] = 1.0  # Unit quaternion

        # Small randomization of initial base position and velocities for exploration
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        self.data.qpos[7:] += rng.normal(0.0, 0.02, size=self.model.nq - 7)
        self.data.qvel[:] += rng.normal(0.0, 0.02, size=self.model.nv)

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self.applied_torque = np.zeros(self.model.nu, dtype=np.float32)
        self.left_air_time = 0.0
        self.right_air_time = 0.0
        self.left_contact_history = False
        self.right_contact_history = False
        self.last_touchdown_foot = None

        obs = self._get_obs()
        return obs, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
