"""
Stable-Baselines3 Callbacks for Unitree H1 Training.

Includes:
  - VelocityCurriculumCallback: Ramps target forward velocity from initial to target speed.
  - BiomechanicsTensorboardCallback: Logs detailed reward breakdowns and survival stats.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class VelocityCurriculumCallback(BaseCallback):
    """
    Gradually ramps target forward velocity from `initial_vel` to `target_vel`
    over `ramp_timesteps` to help the policy stabilize standing before full-speed walking.
    """
    def __init__(
        self,
        initial_vel: float = 0.3,
        target_vel: float = 1.0,
        ramp_timesteps: int = 500_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.initial_vel = initial_vel
        self.target_vel = target_vel
        self.ramp_timesteps = ramp_timesteps
        self.current_target_vel = initial_vel

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.ramp_timesteps))
        self.current_target_vel = self.initial_vel + progress * (self.target_vel - self.initial_vel)

        # Update target_velocity across all parallel vectorized environments
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_target_velocity", self.current_target_vel)

        if self.n_calls % 10_000 == 0:
            self.logger.record("curriculum/target_velocity", self.current_target_vel)
        return True


class BiomechanicsTensorboardCallback(BaseCallback):
    """
    Extracts detailed reward breakdowns and telemetry from env info dictionaries
    and logs them as scalar summaries in TensorBoard.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        # Extract info from the first sub-environment
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        info = infos[0]
        if "reward_breakdown" in info:
            for k, v in info["reward_breakdown"].items():
                self.logger.record(f"reward_components/{k}", float(v))

        if "telemetry" in info:
            for k, v in info["telemetry"].items():
                self.logger.record(f"telemetry/{k}", float(v))

        return True
