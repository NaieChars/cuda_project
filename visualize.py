import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

with open("particles_output.bin", "rb") as f:
    n, frame_count = np.fromfile(f, dtype=np.int32, count=2)
    n = int(n)                      
    frame_count = int(frame_count)  
    box_min_x, box_max_x, box_min_y, box_max_y = np.fromfile(f, dtype=np.float32, count=4)
    data = np.fromfile(f, dtype=np.float32)

data = data.reshape(frame_count, n, 2)  # [frame][particle][x, y]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(box_min_x, box_max_x)
ax.set_ylim(box_min_y, box_max_y)
ax.set_aspect('equal')
scat = ax.scatter(data[0, :, 0], data[0, :, 1], s=3, c='steelblue')
title = ax.set_title("Frame 0")

def update(frame):
    scat.set_offsets(data[frame])
    title.set_text(f"Frame {frame}")
    return scat, title

ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=1000 / 60, blit=False)
plt.show()