# MHD Magnetic Reconnection Simulator

## Overview

This project simulates **magnetic reconnection** in a Harris current sheet using **2D resistive magnetohydrodynamics (MHD)**. Magnetic reconnection is a fundamental plasma physics process where magnetic field lines break and reconnect, converting magnetic energy into kinetic and thermal energy. This phenomenon is crucial in understanding solar flares, Earth's magnetosphere, and laboratory plasma experiments.

## Project Structure

```
Solver/
├── app.py                    # Interactive GUI for real-time simulation
├── analysis_runner.py        # Automated parameter studies and analysis
├── mhd_solver.py            # Core MHD equations solver (Rusanov scheme + RK4)
├── initial_cond.py          # Harris sheet initial conditions
├── physical_test.py         # Diagnostic functions (energy, div B, currents)
├── config.py                # Global simulation parameters
└── Harris_sheet_plot.py     # Visualization of Harris sheet profiles
```

## Physics Background

### Harris Current Sheet
The simulation uses a **Harris sheet** equilibrium configuration:
- Magnetic field reverses direction across a thin current layer
- Creates conditions favorable for magnetic reconnection
- Characterized by:
  - **B₀**: Magnetic field strength
  - **a**: Current sheet thickness (0.05 by default)
  - Hyperbolic tangent profile: `Bₓ = B₀ tanh(y/a)`

### Governing Equations
Solves the 2D resistive MHD equations:
- **Mass conservation**: ∂ρ/∂t + ∇·(ρv) = 0
- **Momentum**: ∂(ρv)/∂t + ∇·(ρvv + pI + BB - B²I/2) = 0
- **Magnetic induction**: ∂B/∂t - ∇×(v×B) = η∇²B (with resistivity)
- **Energy**: Includes kinetic, magnetic, and internal energy with Ohmic heating

### Numerical Method
- **Spatial discretization**: Rusanov (local Lax-Friedrichs) finite volume scheme
- **Time integration**: 4th-order Runge-Kutta (RK4)
- **Boundary conditions**: Periodic in both x and y
- **Stability**: CFL condition for hyperbolic + diffusive timestep constraints

## Features

### 1. Interactive GUI (`app.py`)

Real-time simulation with visual feedback and parameter control.

#### Controls
- **Magnetic Field (B₀)**: 0.1 - 2.0 (controls reconnection strength)
- **Perturbation**: 0.01 - 0.5 (triggers reconnection instability)
- **Resistivity (η)**: 0.001 - 0.1 (controls reconnection rate)
- **Simulation Speed**: 1-50 (adjusts timesteps per frame)
- **Start/Pause/Reset**: Full simulation control

#### Visualizations
1. **Current Density (Jz)**: Shows reconnection X-points and current sheets
2. **Magnetic Field**: Vector field with magnitude coloring

#### Real-Time Diagnostics
- Total, kinetic, magnetic, and internal energies
- Maximum |div B| (should remain small for physical validity)
- Electric field at center (Ez)
- Grid information and timestep

#### Running the GUI
```python
python app.py
```

### 2. Automated Analysis (`analysis_runner.py`)

Systematic parameter studies with quantitative metrics.

#### Available Studies

**Study Types:**
1. **B₀ Study**: Magnetic field strength (0.5, 0.6, 0.7, 0.8)
2. **η Study**: Resistivity (0.0001, 0.0005, 0.001, 0.005)
3. **Perturbation Study**: Initial perturbation (0.01, 0.05, 0.1, 0.2)
4. **Resolution Study**: Grid sizes (64×64, 128×128, 256×256)

#### Metrics Computed
- **Reconnection rate**: Ratio of inflow to Alfvén velocity
- **Maximum current density**: Peak Jz value
- **Energy evolution**: Total, kinetic, magnetic, internal
- **Electric field**: Ez at reconnection site
- **Conservation errors**: Energy and div B violations

#### Running Analysis
```python
python analysis_runner.py
```

**Interactive Menu:**
```
1. Run single simulation
2. Run B0 parameter study
3. Run resistivity (eta) study
4. Run perturbation study
5. Run resolution study
6. Run all studies
```

**Outputs:**
- CSV files with time-series data for each simulation
- Summary statistics for parameter studies
- Automated plots comparing parameter effects:
  - Max reconnection rate vs parameter
  - Magnetic energy loss vs parameter
  - Energy conservation error vs parameter
  - Div(B) error vs parameter


### 3. Supporting Modules

#### `mhd_solver.py`
Core numerical solver implementing:
- Grid initialization
- Primitive ↔ conserved variable conversion
- Rusanov flux computation with interface reconstruction
- Resistive diffusion terms
- RK4 time integration with adaptive timestep

#### `initial_cond.py`
Harris sheet setup with:
- Pressure balance: `p = p₀ + B₀²/2 (1 - tanh²(y/a))`
- Density profile for equilibrium
- Small By perturbation: `δBy ∝ sin(2πx/Lx) exp(-(y-Ly/2)²/a²)`

#### `physical_test.py`
Diagnostic functions:
- `Jz_component()`: Current density from curl of B
- `DivB()`: Checks magnetic field divergence (should be ~0)
- `Total_Energy()`: Integrates energy densities
- `Electric_Field()`: Computes Ez at reconnection site

#### `config.py`
Global parameters:
- Grid: 128×128, domain 1.0×0.5
- γ = 5/3 (adiabatic index)
- CFL = 0.1 (conservative for stability)
- Safety limits for density/pressure

## Installation Requirements

```bash
pip install numpy matplotlib scipy pandas seaborn tkinter
```

## Expected Physical Behavior

### Key Signatures
- **Reconnection rate**: Typically 0.05-0.2 (Sweet-Parker to fast reconnection)
- **Energy conversion**: 20-40% of magnetic energy → kinetic + thermal
- **Current concentration**: Jz peaks at X-points
- **Electric field**: Ez spikes during active reconnection

## Validation Checks

Good simulation should show:
- ✅ Div(B) < 10⁻⁶ (magnetic field divergence-free)
- ✅ Energy conservation error < 1% (without strong resistivity)
- ✅ Reasonable reconnection rates (0.01-0.3)
- ✅ Smooth field evolution (no numerical oscillations)

## Tips for Best Results

1. **Start with defaults**: B₀=1.0, pert=0.1, η=0.01
2. **Higher resistivity**: Faster reconnection but less physical
3. **Resolution**: 128×128 adequate, 256×256 for publication quality
4. **Long simulations**: Run to t=1.0 or until equilibrium
5. **Parameter studies**: Use logarithmic spacing for η

## Scientific Applications

This code can investigate:
- **Sweet-Parker vs fast reconnection**: By varying resistivity
- **Plasmoid instability**: At low resistivity and high resolution
- **Parameter scaling**: How reconnection rate depends on B₀, η, perturbation
- **Energy partition**: Conversion efficiency of magnetic energy

## Limitations

- 2D only (no 3D effects)
- Resistive MHD (no kinetic effects, Hall term, or electron physics)
- Ideal gas equation of state
- Periodic boundaries (limits very large domains)
- Single fluid approximation

## References

For background on magnetic reconnection and numerical MHD:
- Priest & Forbes, "Magnetic Reconnection" (2000)
- Birn & Priest, "Reconnection of Magnetic Fields" (2007)
- Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2009)
