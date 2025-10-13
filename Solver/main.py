import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from mhd_solver import init_grid, RK4, conserved_to_primitives
from initial_cond import Harris_sheet
from physical_test import Jz_component

# --- Initialize grid and solution ---
x, y, dx, dy, X, Y = init_grid()
U = Harris_sheet()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Initial plots ---
Jz = Jz_component(U)
rho, vx, vy, Bx, By, p = conserved_to_primitives(U)

cont1 = ax1.contourf(X, Y, gaussian_filter(Jz, sigma=1), levels=50, cmap='seismic')
ax1.set_title('Jz (current density)')
fig.colorbar(cont1, ax=ax1)

strm = ax2.streamplot(X, Y, Bx, By, color=np.sqrt(Bx**2 + By**2), cmap='plasma', linewidth=1.5)
ax2.set_title('Magnetic field')

# --- Animation update function ---
def update(frame):
    global U
    # Advance solution
    U, dt = RK4(U)

    # Recompute diagnostics
    Jz = Jz_component(U)
    rho, vx, vy, Bx, By, p = conserved_to_primitives(U)

    # Clear axes
    ax1.clear()
    ax2.clear()

    # Update Jz contour
    ax1.contourf(X, Y, gaussian_filter(Jz, sigma=1), levels=50, cmap='seismic')
    ax1.set_title(f'Jz (t = {frame*dt:.3f})')

    # Update magnetic field streamplot
    strm = ax2.streamplot(X, Y, Bx, By, color=np.sqrt(Bx**2 + By**2), cmap='plasma', linewidth=1.5)
    ax2.set_title('Magnetic field')

    return ax1, ax2

ani = FuncAnimation(fig, update, frames=300, interval=50, blit=False)
plt.show()