#Initialization of the grid

#Number of Points
Nx = 128
Ny = 128

#Length of the Grid
Lx = 1.0
Ly = 0.5

#Cell Dimension
dx = Lx/Nx
dy = Ly/Ny

#Initialization of the Physical Constants

#Adiabatic Factor
gamma = 5.0/3.0 #monoatomic

#Resistivity
ETA = 0.01

#Time control parameters
CFL = 0.3
output_interval = 70
tmax = 1.0

#For Safety(Optional)
min_density = 1e-4
min_pressure = 1e-6
