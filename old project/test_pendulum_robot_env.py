import matplotlib.pyplot as plt
import numpy as np
from pendulum_robot_env import SphericalRobotEnv

# Initialize environment
env = SphericalRobotEnv()

# Constant action (dot_beta_cmd = 1.0 rad/s)
constant_action = np.array([50.0], dtype=np.float32)

# Storage for plotting
phi_list = []
beta_list = []
x_list = []
t_list = []

# Reset env
obs = env.reset()
steps = 5000  # simulate for 5 seconds at dt=0.01

for i in range(steps):
    obs, reward, done, _ = env.step(constant_action)
    # Extract full internal state
    beta, beta_dot, phi, phi_dot, x = env.state
    t = i * env.dt

    beta_list.append(beta)
    phi_list.append(phi)
    x_list.append(x)
    t_list.append(t)

# Plot results and save to PNG
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t_list, beta_list)
plt.ylabel("β (rad)")
plt.title("Pendulum-Driven Spherical Robot | Constant β̇_cmd")

plt.subplot(3, 1, 2)
plt.plot(t_list, phi_list)
plt.ylabel("ϕ (rad)")

plt.subplot(3, 1, 3)
plt.plot(t_list, x_list)
plt.ylabel("x (m)")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.savefig("pendulum_robot_constant_beta.png")  # Save as PNG
print("Saved to pendulum_robot_constant_beta.png")
