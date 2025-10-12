import numpy as np
import matplotlib.pyplot as plt
import config

#Initialization of the Grid Space
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
        dx,dy: cell dimensions
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

#The np.roll helps me to introduce the periodic boundary conditions, 
#without explicitly establishing a function for them. Therefore, it is more efficient.

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
        rho (_type_): density
        vx (_type_): x-velocity
        vy (_type_): y-velocity
        Bx (_type_): x-Magnetic Field
        By (_type_): y-Magnetic Field
        p (_type_): pressure

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
        U (..., 6): array of conserved variables [rho, rho*vx, rho*vy, Bx, By, E]\
            
    Returns:
        rho, vx, vy, Bx, By, p : arrays of primitive variables
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





