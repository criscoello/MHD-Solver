import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
import tkinter as tk
from tkinter import ttk
import threading
import sys
import os

# Add the Solver directory to the path
solver_path = os.path.join(os.path.dirname(__file__), '..', 'Solver')
sys.path.insert(0, os.path.abspath(solver_path))

from mhd_solver import init_grid, RK4, conserved_to_primitives, primitives_to_conserved
from physical_test import Jz_component
import config

class MHDSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MHD Magnetic Reconnection Simulator")
        self.root.geometry("1300x750")
        
        # Initialize simulation
        self.x, self.y, self.dx, self.dy, self.X, self.Y = init_grid()
        
        # Default parameters
        self.B0 = 1.0
        self.pert_amp = 0.1
        self.eta = 0.01
        
        self.U = self.create_harris_sheet()
        self.current_time = 0.0
        self.is_running = False
        self.is_paused = False
        
        # Create UI
        self.create_widgets()
        
    def create_harris_sheet(self):
        """Create Harris sheet with current parameters"""
        a = 0.5
        p0 = 0.1
        rho0 = 1.0
        rho1 = 0.0
        
        # Magnetic field: reversed along y
        Bx = self.B0 * np.tanh((self.Y - config.Ly/2) / a)
        By = np.zeros_like(Bx)
        
        # Add small perturbation to trigger reconnection
        By += self.pert_amp * np.sin(2 * np.pi * self.X / config.Lx) * np.exp(-(self.Y - config.Ly/2)**2 / a**2)
        
        # Pressure and density
        p = p0 + 0.5 * self.B0**2 * (1 - np.tanh((self.Y - config.Ly/2) / a)**2)
        rho = rho0 + rho1 * (1 / np.cosh((self.Y - config.Ly/2) / a))**2
        
        # Initial Velocities
        vx = np.zeros_like(Bx)
        vy = np.zeros_like(By)
        
        # Update config with current eta
        config.ETA = self.eta
        
        U = primitives_to_conserved(rho, vx, vy, Bx, By, p)
        return U
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Parameter Controls
        param_frame = ttk.LabelFrame(left_panel, text="Simulation Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # B0 - Magnetic Field Strength
        ttk.Label(param_frame, text="Magnetic Field (B0):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.B0_var = tk.DoubleVar(value=self.B0)
        self.B0_entry = ttk.Entry(param_frame, textvariable=self.B0_var, width=10)
        self.B0_entry.grid(row=0, column=1, pady=5, padx=5)
        ttk.Label(param_frame, text="(0.1 - 2.0)").grid(row=0, column=2, sticky=tk.W)
        
        # Perturbation Amplitude
        ttk.Label(param_frame, text="Perturbation (pert_amp):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.pert_var = tk.DoubleVar(value=self.pert_amp)
        self.pert_entry = ttk.Entry(param_frame, textvariable=self.pert_var, width=10)
        self.pert_entry.grid(row=1, column=1, pady=5, padx=5)
        ttk.Label(param_frame, text="(0.01 - 0.5)").grid(row=1, column=2, sticky=tk.W)
        
        # Resistivity (ETA)
        ttk.Label(param_frame, text="Resistivity (η):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.eta_var = tk.DoubleVar(value=self.eta)
        self.eta_entry = ttk.Entry(param_frame, textvariable=self.eta_var, width=10)
        self.eta_entry.grid(row=2, column=1, pady=5, padx=5)
        ttk.Label(param_frame, text="(0.001 - 0.1)").grid(row=2, column=2, sticky=tk.W)
        
        # Apply button
        self.apply_button = ttk.Button(param_frame, text="Apply & Reset", command=self.apply_parameters)
        self.apply_button.grid(row=3, column=0, columnspan=3, pady=10)
        
        # Control Panel
        control_frame = ttk.LabelFrame(left_panel, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        self.start_button = ttk.Button(control_frame, text="▶ Start", command=self.start_simulation, width=15)
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.pause_button = ttk.Button(control_frame, text="⏸ Pause", command=self.pause_simulation, 
                                       state='disabled', width=15)
        self.pause_button.pack(fill=tk.X, pady=2)
        
        self.reset_button = ttk.Button(control_frame, text="↻ Reset", command=self.reset_simulation, width=15)
        self.reset_button.pack(fill=tk.X, pady=2)
        
        # Speed control
        speed_frame = ttk.LabelFrame(left_panel, text="Simulation Speed", padding="10")
        speed_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.speed_var = tk.IntVar(value=10)
        self.speed_scale = ttk.Scale(speed_frame, from_=1, to=50, orient='horizontal', 
                                      variable=self.speed_var, length=180)
        self.speed_scale.pack(fill=tk.X)
        ttk.Label(speed_frame, text="Slow ← → Fast").pack()
        
        # Info Panel
        info_frame = ttk.LabelFrame(left_panel, text="Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.time_label = ttk.Label(info_frame, text="Time: 0.0000", font=('Arial', 10, 'bold'))
        self.time_label.pack(anchor=tk.W, pady=2)
        
        self.status_label = ttk.Label(info_frame, text="Status: Ready", font=('Arial', 10))
        self.status_label.pack(anchor=tk.W, pady=2)
        
        ttk.Separator(info_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self.info_text = tk.Text(info_frame, height=10, width=25, wrap=tk.WORD, 
                                 font=('Arial', 9), state='disabled')
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.update_info_text()
        
        # Right panel for plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 5))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        # Initial plots
        self.update_plots()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_info_text(self):
        """Update the information text box"""
        info = f"""Current Parameters:
        
B₀ = {self.B0:.3f}
Perturbation = {self.pert_amp:.3f}
η (resistivity) = {self.eta:.4f}

Grid: {config.Nx} × {config.Ny}
Domain: {config.Lx} × {config.Ly}
Max Time: {config.tmax}

About:
This simulation models magnetic reconnection in a Harris current sheet using 2D resistive MHD equations."""
        
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        self.info_text.config(state='disabled')
        
    def apply_parameters(self):
        """Apply new parameters and reset simulation"""
        try:
            # Validate and update parameters
            new_B0 = float(self.B0_var.get())
            new_pert = float(self.pert_var.get())
            new_eta = float(self.eta_var.get())
            
            # Basic validation
            if not (0.1 <= new_B0 <= 2.0):
                raise ValueError("B0 must be between 0.1 and 2.0")
            if not (0.01 <= new_pert <= 0.5):
                raise ValueError("Perturbation must be between 0.01 and 0.5")
            if not (0.001 <= new_eta <= 0.1):
                raise ValueError("Resistivity must be between 0.001 and 0.1")
            
            self.B0 = new_B0
            self.pert_amp = new_pert
            self.eta = new_eta
            
            # Reset simulation with new parameters
            self.reset_simulation()
            self.update_info_text()
            
            self.status_label.config(text="Status: Parameters Applied")
            
        except ValueError as e:
            self.status_label.config(text=f"Error: {str(e)}")
        
    def update_plots(self):
        # Compute diagnostics
        Jz = Jz_component(self.U)
        rho, vx, vy, Bx, By, p = conserved_to_primitives(self.U)
        B_magnitude = np.sqrt(Bx**2 + By**2)
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot Jz
        Jz_smooth = gaussian_filter(Jz.T, sigma=1)
        vmax = max(np.max(np.abs(Jz_smooth)), 1e-6)
        
        self.ax1.contourf(self.X.T, self.Y.T, Jz_smooth, levels=50, 
                         cmap='seismic', vmin=-vmax, vmax=vmax)
        self.ax1.set_title(f'Current Density (Jz)', fontsize=11, fontweight='bold')
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.set_aspect('equal')
        
        # Plot magnetic field
        skip = 4
        X_sub = self.X[::skip, ::skip]
        Y_sub = self.Y[::skip, ::skip]
        Bx_sub = Bx[::skip, ::skip]
        By_sub = By[::skip, ::skip]
        B_mag_sub = B_magnitude[::skip, ::skip]
        
        self.ax2.quiver(X_sub.T, Y_sub.T, Bx_sub.T, By_sub.T, B_mag_sub.T,
                       cmap='plasma', scale=20, width=0.003, pivot='mid')
        self.ax2.set_title(f'Magnetic Field', fontsize=11, fontweight='bold')
        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_aspect('equal')
        self.ax2.set_xlim(0, config.Lx)
        self.ax2.set_ylim(0, config.Ly)
        
        self.fig.tight_layout()
        
    def simulation_loop(self):
        while self.is_running:
            if not self.is_paused:
                # Run multiple steps based on speed setting
                steps = self.speed_var.get()
                
                for _ in range(steps):
                    if self.current_time >= config.tmax:
                        self.stop_simulation()
                        return
                    
                    self.U, dt = RK4(self.U)
                    self.current_time += dt
                
                # Update GUI in main thread
                self.root.after(0, self.update_gui)
            
            # Small delay to prevent CPU overload
            threading.Event().wait(0.05)
    
    def update_gui(self):
        self.update_plots()
        self.canvas.draw()
        self.time_label.config(text=f"Time: {self.current_time:.4f}")
        
    def start_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.start_button.config(state='disabled')
            self.pause_button.config(state='normal')
            self.apply_button.config(state='disabled')
            self.B0_entry.config(state='disabled')
            self.pert_entry.config(state='disabled')
            self.eta_entry.config(state='disabled')
            self.status_label.config(text="Status: Running")
            
            # Start simulation in separate thread
            self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
            self.sim_thread.start()
    
    def pause_simulation(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.config(text="▶ Resume")
            self.status_label.config(text="Status: Paused")
        else:
            self.pause_button.config(text="⏸ Pause")
            self.status_label.config(text="Status: Running")
    
    def stop_simulation(self):
        self.is_running = False
        self.is_paused = False
        self.start_button.config(state='normal')
        self.pause_button.config(state='disabled', text="⏸ Pause")
        self.apply_button.config(state='normal')
        self.B0_entry.config(state='normal')
        self.pert_entry.config(state='normal')
        self.eta_entry.config(state='normal')
        self.status_label.config(text="Status: Completed")
    
    def reset_simulation(self):
        self.is_running = False
        self.is_paused = False
        self.U = self.create_harris_sheet()
        self.current_time = 0.0
        
        self.start_button.config(state='normal')
        self.pause_button.config(state='disabled', text="⏸ Pause")
        self.apply_button.config(state='normal')
        self.B0_entry.config(state='normal')
        self.pert_entry.config(state='normal')
        self.eta_entry.config(state='normal')
        self.status_label.config(text="Status: Ready")
        self.time_label.config(text="Time: 0.0000")
        
        self.update_plots()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = MHDSimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()