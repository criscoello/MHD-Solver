import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from mhd_solver import init_grid, RK4, conserved_to_primitives
from initial_cond import Harris_sheet
from physical_test import Jz_component
import config

# --- Initialize grid and solution ---
x, y, dx, dy, X, Y = init_grid()
U = Harris_sheet()

# Time tracking
current_time = 0.0
output_counter = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.1, wspace=0.3)

# --- Initial plots ---
Jz = Jz_component(U)
rho, vx, vy, Bx, By, p = conserved_to_primitives(U)

# Jz contour plot (transpose for correct orientation)
cont1 = ax1.contourf(X.T, Y.T, gaussian_filter(Jz.T, sigma=1), levels=50, cmap='seismic')
ax1.set_title(f'Jz (current density) - t = {current_time:.4f}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
cbar1 = fig.colorbar(cont1, ax=ax1)

# Magnetic field streamplot (use 1D arrays x, y for streamplot)
B_magnitude = np.sqrt(Bx**2 + By**2)
strm = ax2.streamplot(x, y, Bx.T, By.T, color=B_magnitude.T, cmap='plasma', 
                      linewidth=1.5, density=1.5, arrowsize=1.2)
ax2.set_title(f'Magnetic Field - t = {current_time:.4f}')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
cbar2 = fig.colorbar(strm.lines, ax=ax2)

# --- Animation update function ---
def update(frame):
    global U, current_time, output_counter
    
    # Advance solution multiple steps per frame for smoother progression
    steps_per_frame = config.output_interval
    
    for _ in range(steps_per_frame):
        U, dt = RK4(U)
        current_time += dt
        output_counter += 1
        
        # Stop if we've reached max time
        if current_time >= config.tmax:
            break
    
    # Recompute diagnostics
    Jz = Jz_component(U)
    rho, vx, vy, Bx, By, p = conserved_to_primitives(U)
    B_magnitude = np.sqrt(Bx**2 + By**2)
    
    # Clear axes
    ax1.clear()
    ax2.clear()
    
    # Update Jz contour with consistent color scale
    Jz_smooth = gaussian_filter(Jz.T, sigma=1)
    vmax = np.max(np.abs(Jz_smooth))
    cont = ax1.contourf(X.T, Y.T, Jz_smooth, levels=50, cmap='seismic', 
                        vmin=-vmax, vmax=vmax)
    ax1.set_title(f'Jz (current density) - t = {current_time:.4f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Update magnetic field streamplot
    ax2.streamplot(x, y, Bx.T, By.T, color=B_magnitude.T, cmap='plasma', 
                   linewidth=1.5, density=1.5, arrowsize=1.2)
    ax2.set_title(f'Magnetic Field - t = {current_time:.4f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Print progress
    if frame % 10 == 0:
        print(f"Frame {frame}: t = {current_time:.4f}, step = {output_counter}")
    
    return ax1, ax2

# Calculate number of frames needed
total_frames = int(config.tmax / (config.output_interval * 0.001)) + 50

print(f"Starting animation with {total_frames} frames")
print(f"Grid: {config.Nx} x {config.Ny}")
print(f"Domain: {config.Lx} x {config.Ly}")
print(f"Resistivity η = {config.ETA}")

ani = FuncAnimation(fig, update, frames=total_frames, interval=50, 
                    blit=False, repeat=False)

plt.show()