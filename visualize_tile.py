import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

with open("day30_tiled_output.bin", "rb") as f:
    n, frame_count = np.fromfile(f, dtype=np.int32, count=2)
    n = int(n)                      
    frame_count = int(frame_count)
    masses = np.fromfile(f, dtype=np.float32, count=n)
    data = np.fromfile(f, dtype=np.float32)

data = data.reshape(frame_count, n, 2)

# 用质量区分点的大小：中心天体(masses[0]很大)会画得明显更大
sizes = np.clip(masses * 0.05, 1, 100)
colors = np.where(masses > 100, 'orange', 'steelblue')

fig, ax = plt.subplots(figsize=(7, 7))
LIM = 15
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_aspect('equal')
ax.set_facecolor('black')
scat = ax.scatter(data[0, :, 0], data[0, :, 1], s=sizes, c=colors)
title = ax.set_title("Frame 0")

def update(frame):
    scat.set_offsets(data[frame])
    title.set_text(f"Frame {frame}")
    return scat, title

ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=1000/60, blit=False)
plt.show()