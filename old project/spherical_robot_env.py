import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os

class SphericalRobotEnv(gym.Env):
    def __init__(self, model_path="spherical_robot.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.time = 0.0
        self.dt = self.model.opt.timestep

        # Observation: [x, y, z, vx, vy, vz, joint_x, joint_y, joint_vel_x, joint_vel_y]
        obs_high = np.array([np.inf]*10, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Action: torques for joint_x and joint_y in [-1, 1]
        self.action_space = spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.time = 0.0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.data.ctrl[:] = action

        mujoco.mj_step(self.model, self.data)
        self.time += self.dt

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere")]  # [x, y, z]
        vel = self.data.cvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere")][3:]  # linear velocity
        joint_pos = self.data.qpos[-2:]  # joint_x, joint_y
        joint_vel = self.data.qvel[-2:]  # joint_vel_x, joint_vel_y
        return np.concatenate([pos, vel, joint_pos, joint_vel]).astype(np.float32)

    def _get_reward(self):
        desired = self._desired_position(self.time)
        actual = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere")][:2]  # x, y
        error = np.linalg.norm(desired - actual)
        ctrl_penalty = 0.01 * np.sum(np.square(self.data.ctrl))
        return -error - ctrl_penalty

    def _desired_position(self, t):
        # Example: sinusoidal x motion, constant y
        x_des = 0.3 * np.sin(0.5 * t)
        y_des = 0.0
        return np.array([x_des, y_des])

    def render(self):
        mujoco.viewer.launch_passive(self.model, self.data)

    def close(self):
        pass
