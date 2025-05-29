import gym
from gym import spaces
import numpy as np

class SphericalRobotEnv(gym.Env):
    """
    Gym-like environment for a pendulum-driven spherical robot.

    Input: dot_beta_cmd (velocity command to low-pass filter)
    Internal state: [beta, beta_dot, phi, phi_dot, x]
    Observation: [phi, beta]
    """

    def __init__(self, dt=0.01, omega_n=10.0, params=None):
        super().__init__()

        self.dt = dt
        self.omega_n = omega_n

        self.params = params or {
            "m_p": 0.2,         # mass of pendulum
            "r": 0.043,         # length of pendulum
            "rho": 0.15,        # radius of spherical shell
            "M": 3.6,           # total mass (3.4 shell + 0.2 pendulum)
            "I_c": 0.21,    # shell inertia
            "g": 9.81           # gravity
        }

        # Action: dot_beta_cmd
        self.action_space = spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)

        # Observation: [phi, beta]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.reset()

    def step(self, action):
        dot_beta_cmd = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        beta, beta_dot, phi, phi_dot, x = self.state

        # === Low-pass filter dynamics ===
        beta_dd = -self.omega_n * beta_dot + self.omega_n * dot_beta_cmd
        beta_dot += beta_dd * self.dt
        beta += beta_dot * self.dt

        # === Dynamics ===
        p = self.params
        delta = beta - phi

        num = (
            p["m_p"] * (p["r"]**2 - p["rho"] * p["r"] * np.cos(delta)) * beta_dd +
            p["m_p"] * p["rho"] * p["r"] * np.sin(delta) * (beta_dot - phi_dot)**2 +
            p["M"] * p["g"] * p["rho"] * np.sin(delta)
        )
        den = p["I_c"] + p["m_p"] * (p["rho"]**2 + p["r"]**2 - 2 * p["rho"] * p["r"] * np.cos(delta))

        phi_dd = num / den - 2

        # Integrate φ and x
        phi_dot += phi_dd * self.dt
        phi += phi_dot * self.dt
        x += p["rho"] * phi_dot * self.dt

        # Update full state
        self.state = np.array([beta, beta_dot, phi, phi_dot, x], dtype=np.float32)

        # Tracking reward (you can modify this later)
        x_target = 1.0  # Constant target for now
        reward = - (x - x_target)**2

        done = False
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.state = np.zeros(5, dtype=np.float32)  # [beta, beta_dot, phi, phi_dot, x]
        return self._get_obs()

    def _get_obs(self):
        beta, _, phi, _, _ = self.state
        return np.array([phi, beta], dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass

