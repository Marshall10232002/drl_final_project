import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from spherical_robot_env import SphericalRobotEnv

# Initialize environment
env = SphericalRobotEnv("spherical_robot.xml")

# Logging containers
positions = []
pendulum_angles = []

# Run simulation
obs, _ = env.reset()
for step in range(1000):
    # 🔁 Choose one of these:
    action = env.action_space.sample()     # Random torque
    #action = np.array([30.0, 0.0])                  # Constant torque on joint_x

    obs, reward, done, trunc, info = env.step(action)

    # Extract data
    x, y, z = obs[0:3]
    joint_x, joint_y = obs[6:8]
    positions.append([x, y])
    pendulum_angles.append([joint_x, joint_y])

# Convert logs
positions = np.array(positions)
pendulum_angles = np.array(pendulum_angles)

# === Static XY Trajectory Plot ===
plt.figure(figsize=(6, 6))
plt.plot(positions[:, 0], positions[:, 1], label="Sphere Trajectory")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Spherical Robot XY Path")
plt.legend()
plt.grid()
plt.axis("equal")
plt.savefig("trajectory_static.png")
plt.close()

# === Pendulum Angles ===
plt.figure()
plt.plot(pendulum_angles[:, 0], label="Joint X")
plt.plot(pendulum_angles[:, 1], label="Joint Y")
plt.xlabel("Timestep")
plt.ylabel("Angle (rad)")
plt.title("Pendulum Joint Angles")
plt.legend()
plt.grid()
plt.savefig("pendulum_angles.png")
plt.close()

# === Animated Trajectory (MP4) ===
fig, ax = plt.subplots()
ax.set_xlim(np.min(positions[:, 0]) - 0.1, np.max(positions[:, 0]) + 0.1)
ax.set_ylim(np.min(positions[:, 1]) - 0.1, np.max(positions[:, 1]) + 0.1)
ax.set_title("Spherical Robot Trajectory (Animation)")
line, = ax.plot([], [], lw=2)

def animate(i):
    line.set_data(positions[:i+1, 0], positions[:i+1, 1])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(positions), interval=33, blit=True)
ani.save("trajectory.mp4", fps=30, dpi=200)
plt.close()

print("✅ Simulation complete. Plots saved as PNG and MP4.")
