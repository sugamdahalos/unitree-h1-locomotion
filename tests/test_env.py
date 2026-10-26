"""
Unit tests for UnitreeH1Env and configs.
"""

import unittest
import numpy as np
from configs.default import get_config, EXPERIMENTS
from envs.unitree_h1_env import UnitreeH1Env


class TestUnitreeH1Env(unittest.TestCase):
    def test_configs_validity(self):
        """Ensure all experiment configurations load and specify valid parameters."""
        for exp_key in ["exp1", "exp2", "exp3", "exp4a", "exp4b", "exp4c", "exp5"]:
            cfg = get_config(exp_key)
            self.assertTrue(cfg.name.startswith(exp_key[:4]))
            self.assertGreater(cfg.max_episode_steps, 0)
            self.assertEqual(cfg.nominal_com_height, 0.98)

    def test_exp4_rewards(self):
        """Test that exp4a (symmetry), exp4b (clearance), and exp4c (alternation) compute valid rewards."""
        for exp_key, expected_key in [
            ("exp4a", "reward_symmetry"),
            ("exp4b", "reward_clearance"),
            ("exp4c", "reward_alternation"),
        ]:
            cfg = get_config(exp_key)
            env = UnitreeH1Env(config=cfg)
            obs, info = env.reset(seed=42)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertIn("reward_breakdown", info)
            self.assertIn(expected_key, info["reward_breakdown"])
            self.assertTrue(np.isfinite(reward))
            env.close()

    def test_env_initialization(self):
        """Test environment construction, observation/action space dimensions."""
        cfg = get_config("exp1")
        env = UnitreeH1Env(config=cfg)
        self.assertEqual(env.action_space.shape[0], 10)  # 10 leg actuators
        # Obs: 3 (grav) + 3 (lin_vel) + 3 (ang_vel) + 1 (cmd) + 10 (qpos) + 10 (qvel) + 10 (prev_act) = 40
        self.assertEqual(env.observation_space.shape[0], 40)
        env.close()

    def test_env_step_and_rewards(self):
        """Test step execution, finite state output, and reward breakdown dictionary."""
        cfg = get_config("exp2")
        env = UnitreeH1Env(config=cfg)
        obs, info = env.reset(seed=42)

        self.assertEqual(obs.shape, (40,))
        self.assertTrue(np.isfinite(obs).all())

        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(reward))
            self.assertIn("reward_breakdown", info)
            self.assertIn("reward_velocity", info["reward_breakdown"])
            self.assertIn("reward_orientation", info["reward_breakdown"])
            self.assertIn("telemetry", info)
            if terminated or truncated:
                break
        env.close()

    def test_upper_body_pd_holding(self):
        """Verify that upper body actuators receive active PD torques rather than 0 torque."""
        cfg = get_config("exp2")
        env = UnitreeH1Env(config=cfg)
        env.reset(seed=123)

        # Deflect torso joint slightly to test PD response
        torso_qpos_adr = env.upper_qpos_adrs[0]
        env.data.qpos[torso_qpos_adr] = 0.2  # 0.2 rad deflection
        env.step(np.zeros(10, dtype=np.float32))

        torso_act_id = env.upper_actuator_ids[0]
        applied_torso_torque = env.applied_torque[torso_act_id]
        # Restoring torque must be negative (-Kp * 0.2 < 0)
        self.assertLess(applied_torso_torque, 0.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
