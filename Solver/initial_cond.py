import numpy as np
from mhd_solver import init_grid
from mhd_solver import primitives_to_conserved


def Harris_sheet():
    #Import Meshgrid|cells
    x, y, dx, dy, X, Y = init_grid()
    
    #Initialize parameters
    B0 = 1.0
    a = 0.5
    p0 = 0.1
    rho0 = 1.0
    rho1 = 0.0
    
    # Magnetic field: reversed along y
    Bx = B0 * np.tanh((Y - config.Ly/2) / a)
    By = np.zeros_like(Bx)
    
    # Add small perturbation to trigger reconnection
    pert_amp = 0.01
    By += pert_amp * np.sin(2 * np.pi * X / config.Lx) * np.exp(-(Y - config.Ly/2)**2 / a**2)
    
    # Pressure and density
    p = p0 + 0.5 * B0**2 * (1 - np.tanh((Y - config.Ly/2) / a)**2)
    rho = rho0 + rho1 * (1 / np.cosh((Y - config.Ly/2) / a))**2
    
    #Initial Velocities
    vx = np.zeros_like(Bx)
    vy = np.zeros_like(By)

    #The initial U vector from the predefined function
    U = primitives_to_conserved(rho, vx, vy, Bx, By, p)
    
    return U


    
    