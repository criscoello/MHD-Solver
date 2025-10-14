# 2D MHD Magnetic Reconnection Simulator

A Python-based simulation of magnetic reconnection in a Harris current sheet using 2D resistive magnetohydrodynamics (MHD). This code solves the ideal MHD equations with resistive diffusion using a finite volume method with Rusanov flux.

This simulator models the physical process of magnetic reconnection, where oppositely directed magnetic field lines break and reconnect, releasing large amounts of stored magnetic energy. The Harris sheet configuration provides a classic setup for studying this phenomenon, which is fundamental to understanding solar flares, magnetospheric substorms, and plasma dynamics in astrophysical systems.

## Features

- **2D Resistive MHD Solver**: Full implementation of the resistive MHD equations
- **Rusanov (Local Lax-Friedrichs) Flux**: Robust numerical flux for handling discontinuities
- **RK4 Time Integration**: Fourth-order Runge-Kutta for accurate time evolution
- **Harris Sheet Initial Condition**: Classic magnetic reconnection setup
- **Periodic Boundary Conditions**: Implemented via efficient array rolling
- **Adaptive Time Stepping**: CFL condition for both hyperbolic and diffusive constraints
- **Video Export**: Save simulations as high-quality MP4 videos
- **Physical Diagnostics**: Current density (Jz), magnetic field visualization, energy tracking

## Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- FFmpeg (for video export)

## Project Structure

```
Solver/
├── config.py           # Simulation parameters and grid setup
├── mhd_solver.py       # Core MHD solver (flux functions, time integration)
├── initial_cond.py     # Initial condition setup (Harris sheet)
├── physical_test.py    # Diagnostic functions (current, energy, etc.)
└── main.py             # Main execution and visualization
```

## Usage

### Basic Simulation

Run the simulation with default parameters.

This will generate a video file `magnetic_reconnection.mp4` showing the evolution of the current density and magnetic field topology.

### Configuration

Edit `config.py` to modify simulation parameters.

### Custom Initial Conditions

To implement your own initial conditions, create a new function in `initial_cond.py`:

Then update `main.py` to use your function instead of `Harris_sheet()`.

## Physics of the Harris Sheet

The Harris sheet is an equilibrium configuration with:

- **Reversed magnetic field**: Bx changes sign across y = Ly/2
- **Current sheet**: Strong current Jz = ∂Bx/∂y flows in the z-direction
- **Pressure balance**: Total pressure (thermal + magnetic) is constant
- **Perturbation**: Small By perturbation triggers reconnection

The resistivity (η) allows magnetic field lines to break and reconnect, converting magnetic energy into kinetic and thermal energy.

## Output

The simulation produces:

1. **Video file** (`magnetic_reconnection.mp4`):
   - Left panel: Current density Jz showing X-point formation
   - Right panel: Magnetic field streamlines showing reconnection topology

## Numerical Method

- **Spatial discretization**: Finite volume method with cell-centered values
- **Flux calculation**: Rusanov (Local Lax-Friedrichs) scheme
- **Time integration**: 4th-order Runge-Kutta (RK4)
- **Boundary conditions**: Periodic in both x and y directions
- **Divergence control**: ∇·B maintained through appropriate flux formulation
- **Stability**: CFL condition for hyperbolic terms, diffusive limit for resistive terms

## References

1. Harris, E. G. (1962). "On a plasma sheath separating regions of oppositely directed magnetic field." *Nuovo Cimento*, 23(1), 115-121.
2. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.
3. Biskamp, D. (2000). *Magnetic Reconnection in Plasmas*. Cambridge University Press.

## Author

Cristian Coello - Computational Physics, ELTE
