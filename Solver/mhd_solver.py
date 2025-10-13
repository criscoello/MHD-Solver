import numpy as np
import matplotlib.pyplot as plt
import config

#Initialization of the Grid Space(Discretization)
def init_grid(Nx = config.Nx, Ny = config.Ny, Lx = config.Lx, Ly = config.Ly):
    """
    Initializes the space grid

    Args:
        Nx (integer): Number of divisions in x
        Ny (integer): Number of divisions in y
        Lx (float): Length of x 
        Ly (float): Length of y
        
    Returns:
        x,y: 1D arrays(self-centered coordinates)
        X,Y: 2D meshgrid array(size Nx,Ny)
        dx,dy: cell dimensions - The horizontal and vertical length
    """
    
    #1D arrays for x and y
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    
    #2D meshgrid array (x,y) points
    X,Y = np.meshgrid(x,y, indexing='ij')
    
    #Cell Dimensions
    dx = Lx/Nx
    dy = Ly/Ny
    
    return x, y, dx, dy, X, Y

#Derivative Operators
#This two helps us find the Jz and Div B
def ddx(f, dx):
    """
    Central difference derivative in x with periodic boundaries
    using finite differences.
    
    """
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)

def ddy(f, dy):
    """
    Central difference derivative in y with periodic boundaries
    using finite differences
    
    """
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)

#The np.roll method helps me to introduce the periodic boundary conditions, 
#without explicitly establishing a function for them. Therefore, it is more efficient.
#It rolls back and forth the values inside the arrays

def laplacian(f, dx, dy):
    """
    2D Laplacian with periodic boundaries using finite differences
    
    """
    return (
        (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2
      + (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dy**2
    )

#Change from primitive variables to conserved variables in vector U
def primitives_to_conserved(rho, vx, vy, Bx, By, p):
    """Creating a meshgrid for U, so it takes value for each cell i, and returns
    some specific physical qunatity at that specific point (eg. pressure p, density rho)
    

    Args:
        rho : density
        vx : x-velocity
        vy : y-velocity
        Bx : x-Magnetic Field
        By : y-Magnetic Field
        p  : pressure

    Returns:
        U: The vector variable that contains the "conserved" quantities
    """
    
    U = np.zeros((rho.shape[0], rho.shape[1], 6))
    U[...,0] = rho
    U[...,1] = rho*vx
    U[...,2] = rho*vy
    U[...,3] = Bx
    U[...,4] = By
    E = (p/(config.gamma-1)) + 0.5*rho*(vx**2 + vy**2) + 0.5*(Bx**2 + By**2)
    U[...,5] = E
    
   
    return U
    
#Change from conserved to primitive variables using U vector
def conserved_to_primitives(U):
    """Converts conserved variables U to primitive variables.
    
    Args:
        U (..., 6): array of conserved variables [rho, rho*vx, rho*vy, Bx, By, E]
            
    Returns:
        rho, vx, vy, Bx, By, p : array of primitive variables
    """
    #convert the U vector components into the physical variables
    rho = np.maximum(U[...,0],config.min_density)
    vx = U[...,1]/rho
    vy = U[...,2]/rho
    Bx = U[...,3]
    By = U[...,4]
    KE = 0.5*rho*(vx**2 + vy**2)
    ME = 0.5*(Bx**2 + By**2)
    p = (config.gamma - 1) * (U[...,5] - KE - ME)
    p = np.maximum(p, config.min_pressure)
    
    #If pressure is negative then take the preconfigured min pressure.
    condition = (U[...,5] < (KE + ME))
    p[condition] = config.min_pressure

    return rho, vx, vy, Bx, By, p

def Flux_x(U):
    """This F vector shows you how the "conserved" quantities flow. Fx(U).
    The flux of U in the x direction

    Args:
        U : vector of conserved quantities

    Returns:
        F: The F vector(x-Flux of U)
    """
    rho, vx, vy, bx, by, p = conserved_to_primitives(U)
    
    #Create a similar meshgrid like U vector
    F = np.zeros_like(U)
    
    #Initialize the components of F(U)
    F[...,0] = rho * vx
    F[...,1] = rho*vx*vx + p + 0.5*(bx**2 + by**2) - bx*bx
    F[...,2] = rho*vx*vy - bx*by
    F[...,3] = 0.0
    F[...,4] = vx*by - vy*bx
    F[...,5] = (U[...,5] + p + 0.5*(bx**2 + by**2)) * vx - (bx * (vx*bx + vy*by))
    
    return F

def Flux_y(U):
    """This G vector shows you how the "conserved" quantities flow. Gy(U).
    The flux of U in the y direction

    Args:
        U : vector of conserved quantities

    Returns:
        G: The G vector(y-Flux of U)
    """
    rho, vx, vy, bx, by, p = conserved_to_primitives(U)
    
    #Create a similar meshgrid like U vector
    G = np.zeros_like(U)
    
    #Initialize the components of G(U)
    G[...,0] = rho * vy
    G[...,1] = rho*vy*vx - by*bx
    G[...,2] = rho*vy*vy + p + 0.5*(bx**2 + by**2) - by*by
    G[...,3] = vy*bx - vx*by
    G[...,4] = 0.0
    G[...,5] = (U[...,5] + p + 0.5*(bx**2 + by**2)) * vy - (by * (vx*bx + vy*by))
   
    return G

def Rusanov_Interface_x(U):
    
    # Build left/right states for x-interfaces: shapes (nx+1, ny, 6)
    ULx = np.zeros((config.Nx+1, config.Ny, 6))
    URx = np.zeros((config.Nx+1, config.Ny, 6))
    
    # interior interfaces: left cell i-1, right cell i
    ULx[1:-1, :, :] = U[0:-1, :, :]
    URx[1:-1, :, :] = U[1:, :, :]
    
    #Periodic Boundary conditions
    ULx[0, :, :] = U[-1, :, :]
    URx[0, :, :] = U[0, :, :]
    ULx[-1, :, :] = U[-1, :, :]
    URx[-1, :, :] = U[0, :, :]

    return ULx, URx

def Rusanov_Interface_y(U):
    
    # Build left/right states for y-interfaces: shapes (nx+1, ny, 6)
    ULy = np.zeros((config.Nx, config.Ny+1, 6))
    URy = np.zeros((config.Nx, config.Ny+1, 6))
    
    # interior interfaces: left cell j-1, right cell j
    ULy[:, 1:-1, :] = U[:, :-1, :]
    URy[:, 1:-1, :] = U[:, 1:, :]
    
    #Periodic Boundary Conditions
    ULy[:, 0, :] = U[:, -1, :]
    URy[:, 0, :] = U[:, 0, :]
    ULy[:, -1, :] = U[:, -1, :]
    URy[:, -1, :] = U[:, 0, :]
    
    return ULy, URy


def max_signal_speed(UL, UR, direction='x'):
    rho_L, vx_L, vy_L, bx_L, by_L, p_L = conserved_to_primitives(UL)
    rho_R, vx_R, vy_R, bx_R, by_R, p_R = conserved_to_primitives(UR)

    cs_L = np.sqrt(config.gamma * p_L / rho_L)
    cs_R = np.sqrt(config.gamma * p_R / rho_R)

    vA_L = np.sqrt(bx_L**2 + by_L**2) / np.sqrt(rho_L)
    vA_R = np.sqrt(bx_R**2 + by_R**2) / np.sqrt(rho_R)

    cf_L = np.sqrt(cs_L**2 + vA_L**2)
    cf_R = np.sqrt(cs_R**2 + vA_R**2)

    if direction=='x':
        alpha_L = np.abs(vx_L) + cf_L
        alpha_R = np.abs(vx_R) + cf_R
    elif direction=='y':
        alpha_L = np.abs(vy_L) + cf_L
        alpha_R = np.abs(vy_R) + cf_R

    alpha_interface = np.maximum(alpha_L, alpha_R)
    return alpha_interface
    
def Rusanov_flux_x(U):
    
    ULx, URx = Rusanov_Interface_x(U)
    
    F_L = Flux_x(ULx)
    F_R = Flux_x(URx)
    
    alpha_interface = max_signal_speed(ULx, URx, direction='x')
    alpha_broadcast = alpha_interface[..., None]
    
    Fx = 0.5*(F_L + F_R) - 0.5*alpha_broadcast*(URx - ULx)
    
    RHS_x = - (Fx[1:,:,:] - Fx[:-1,:,:]) / config.dx
    
    return Fx, RHS_x

def Rusanov_flux_y(U):
    
    ULy, URy = Rusanov_Interface_y(U)
    
    F_L = Flux_y(ULy)
    F_R = Flux_y(URy)
    
    alpha_interface = max_signal_speed(ULy, URy, direction='y')
    alpha_broadcast = alpha_interface[..., None]
    
    Fy = 0.5*(F_L + F_R) - 0.5*alpha_broadcast*(URy - ULy)
    
    RHS_y = - (Fy[:,1:,:] - Fy[:,:-1,:]) / config.dy
    
    return Fy, RHS_y

def full_RHS(U):
    Fx, RHS_x = Rusanov_flux_x(U)
    Fy, RHS_y = Rusanov_flux_y(U)
    
    RHS_total = RHS_x + RHS_y

    # Include Resistive diffusion
    eta = config.ETA
    Bx = U[...,3]
    By = U[...,4]

    lap_Bx = laplacian(Bx, config.dx, config.dy)
    lap_By = laplacian(By, config.dx, config.dy)

    RHS_total[...,3] += eta * lap_Bx
    RHS_total[...,4] += eta * lap_By
    
    # For 2D (x,y) with B=(Bx,By), Jz = d_x B_y - d_y B_x
    Jz = ddx(By, config.dx) - ddy(Bx, config.dy)

    # Add resistive(Ohmic) heating to energy equation
    RHS_total[...,5] += config.ETA * (Jz**2)    

    return RHS_total
 
#Time Evolution of U
#Runge-Kutta algorithm
def RK4(U):
    ULx, URx = Rusanov_Interface_x(U)
    ULy, URy = Rusanov_Interface_y(U)
    
    alpha_x = np.max(max_signal_speed(ULx, URx, direction='x'))
    alpha_y = np.max(max_signal_speed(ULy, URy, direction='y'))
    alpha_max = max(alpha_x, alpha_y)
    
    if alpha_max < 1e-8:
        alpha_max = 1e-8
    
    dt_hyperbolic = config.CFL * min(config.dx, config.dy) / alpha_max
    
    if config.ETA > 0.0:
        dt_diffusive = 0.5 * (min(config.dx, config.dy)**2) / config.ETA
    else:
        dt_diffusive = np.inf
        
    dt = min(dt_hyperbolic, dt_diffusive)

    k1 = full_RHS(U)
    k2 = full_RHS(U + 0.5 * dt * k1)
    k3 = full_RHS(U + 0.5 * dt * k2)
    k4 = full_RHS(U + dt * k3)

    U_new = U + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return U_new, dt
    

    
    
    





