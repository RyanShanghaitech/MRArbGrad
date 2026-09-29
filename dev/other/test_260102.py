import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

def get_coprime(n):
    # We want a large stride that is coprime to n
    # A simple way is to pick a large prime or use the Golden Ratio
    # Let's use a large prime near n * 0.618
    stride = int(n * 3.8196601125010510e-1)
    while math.gcd(stride, n) != 1:
        stride += 1
    return stride

def animate_scattered_fill(width=40, height=40):
    num_points = width * height
    stride = get_coprime(num_points)
    
    # Generate the sequence of indices
    # j = (i * stride) % num_points
    indices = [(i * stride) % num_points for i in range(num_points)]
    
    # Convert flat indices to (x, y)
    shuffled_coords = [(idx % width, idx // width) for idx in indices]

    # Setup the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, width)
    ax.set_ylim(-1, height)
    ax.set_title(f"Coprime Stride Filling (Stride: {stride})")
    
    scatter = ax.scatter([], [], marker='s', s=40, c='firebrick', edgecolors='none')
    
    x_data, y_data = [], []

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return scatter,

    def update(frame):
        # Plotting multiple points per frame for a smooth but fast visual
        points_per_frame = 8 
        start_idx = frame * points_per_frame
        end_idx = min(start_idx + points_per_frame, num_points)
        
        for i in range(start_idx, end_idx):
            px, py = shuffled_coords[i]
            x_data.append(px)
            y_data.append(py)
        
        scatter.set_offsets(np.c_[x_data, y_data])
        return scatter,

    total_frames = int(np.ceil(num_points / 8))

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, init_func=init, 
        blit=True, interval=15, repeat=False
    )

    plt.show()

if __name__ == "__main__":
    animate_scattered_fill(40, 40)