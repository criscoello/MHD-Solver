import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import gaussian_filter
from mhd_solver import init_grid, RK4, conserved_to_primitives
from initial_cond import Harris_sheet
from physical_test import Jz_component
import config

# Initialize grid
x, y, dx, dy, X, Y = init_grid()

#Initial conditions
U = Harris_sheet()

# Time evolution counters
t = 0.0 #current time
output_counter = 0

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.1, wspace=0.3)

# Print elementary details about our working space, and parameters
print(f"Grid: {config.Nx} x {config.Ny}")
print(f"Domain: {config.Lx} x {config.Ly}")
print(f"Resistivity η = {config.ETA}")
print(f"Max time: {config.tmax}")
print("\nStarting simulation and video generation...")

# Update function
def update(frame):
    global U, t, output_counter
    
    # Advance solution multiple steps per frame
    steps_per_frame = config.output_interval
    
    for _ in range(steps_per_frame):
        U, dt = RK4(U)
        t += dt
        output_counter += 1
        
        # Stop if we've reached max time
        if t >= config.tmax:
            break
    
    # Recompute diagnostics
    Jz = Jz_component(U)
    rho, vx, vy, Bx, By, p = conserved_to_primitives(U)
    B_magnitude = np.sqrt(Bx**2 + By**2)
    
    # Clear axes
    ax1.clear()
    ax2.clear()
    
    # Jz ContourPlot
    Jz_smooth = gaussian_filter(Jz.T, sigma=1)
    vmax = np.max(np.abs(Jz_smooth))
    cont = ax1.contourf(X.T, Y.T, Jz_smooth, levels=50, cmap='seismic', 
                        vmin=-vmax, vmax=vmax)
    ax1.set_title(f'Jz (current density) - t = {t:.4f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Magentic Field Streamplot
    ax2.streamplot(x, y, Bx.T, By.T, color=B_magnitude.T, cmap='plasma', 
                   linewidth=1.5, density=1.5, arrowsize=1.2)
    ax2.set_title(f'Magnetic Field - t = {t:.4f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Print progress
    if frame % 10 == 0:
        progress = (t / config.tmax) * 100
        print(f"Frame {frame}: t = {t:.4f} ({progress:.1f}% complete)")
        print("Do not close the program")
    
    return ax1, ax2

# Calculate number of frames needed
total_frames = int(config.tmax / (config.output_interval * 0.001)) + 50

print(f"\nGenerating {total_frames} frames...")
print("This may take several minutes depending on your system...\n")

# Create animation
ani = FuncAnimation(fig, update, frames=total_frames, interval=50, 
                    blit=False, repeat=False)

# Set up the video writer
writer = FFMpegWriter(fps=18,  # frames per second
                     metadata=dict(artist='MHD Simulation', 
                                   title='Harris Sheet Magnetic Reconnection'),
                     bitrate=3000)  # Higher bitrate = better quality

# Save the animation as MP4
output_filename = "magnetic_reconnection.mp4"
#Change the name if you do not want to replace the last file
print(f"Saving video to: {output_filename}")
print("Please wait, this will take a few minutes...\n")


#Handling Exceptions
try:
    ani.save(output_filename, writer=writer, dpi=90)
    print(f"\n✓ Video saved successfully as '{output_filename}'!")
    print(f"  Total frames: {total_frames}")
    print(f"  Simulation time: {t:.4f}")
    print(f"  Video duration: {total_frames/20:.1f} seconds")
except Exception as e:
    # You need FFmpeg codec if you want to watch the video
    print(f"\n✗ Error saving video: {e}")
    print("\nNote: This requires FFmpeg to be installed on your system.")
    print("\nAlternatively, try displaying the animation with plt.show() instead.")

plt.close()
#plt.show() if video does not work
print("\nDone!")