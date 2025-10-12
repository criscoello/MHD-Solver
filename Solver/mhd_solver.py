import numpy as np
import matplotlib.pyplot as plt
import config


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
    
    return x,y,X,Y
