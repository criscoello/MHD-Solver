import numpy as np
from mhd_solver import ddx, ddy, conserved_to_primitives
import config

def Jz_component(U):
    
    Bx = U[..., 3]
    By = U[..., 4]
    
    Jz = ddx(By, config.dx) - ddy(Bx, config.dy)
    return Jz

#The Div of B must be Zero (Div B = 0)
def DivB(U):
    
    Bx = U[..., 3]
    By = U[..., 4]
    
    divB = ddx(Bx, config.dx) + ddy(By, config.dy)
    return divB

#Calculating the total energy of the system, and the diverse energies involved
def Total_Energy(U):
    
    rho, vx, vy, Bx, By, p = conserved_to_primitives(U)
    E = U[...,5]
    
    #Energy densities(arrays)
    Kinetic = 0.5*rho*(vx**2 + vy**2)
    Magnetic = 0.5*(Bx**2 + By**2)
    Internal = E - Kinetic - Magnetic
    
    #Integration of energy densities over each cell
    cell_area = config.dx*config.dy
    E_total = np.sum(E)*cell_area
    E_kinetic = np.sum(Kinetic)*cell_area
    E_magnetic = np.sum(Magnetic)*cell_area
    E_internal = np.sum(Internal)*cell_area
    
    return E_total, E_kinetic, E_magnetic, E_internal

def Electric_Field(U):
    
    rho  = U[..., 0]
    mx   = U[..., 1]
    my   = U[..., 2]
    Bx   = U[..., 3]
    By   = U[..., 4]

    vx = mx / rho
    vy = my / rho
    Jz = Jz_component(U)

    Ez = config.ETA * Jz - (vx * By - vy * Bx)

    # Choose the center
    ix = config.Nx // 2
    iy = config.Ny // 2
    return Ez[ix, iy]




        
    