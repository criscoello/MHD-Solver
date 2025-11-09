import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import config
from initial_cond import Harris_sheet
from mhd_solver import RK4, conserved_to_primitives
from physical_test import Total_Energy, DivB, Electric_Field, Jz_component

class AnalysisRunner:
    def __init__(self):
        self.results_dir = "simulation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run_single_simulation(self, B0=1.0, pert_amp=0.1, eta=0.01, 
                            resolution=None, tmax=1.0, log_interval=0.01):
        """Run a single simulation with given parameters and log data"""
        
        # Override config if different resolution is specified
        if resolution:
            original_Nx, original_Ny = config.Nx, config.Ny
            config.Nx, config.Ny = resolution
            config.dx = config.Lx / config.Nx
            config.dy = config.Ly / config.Ny
        
        # Set parameters
        original_eta = config.ETA
        config.ETA = eta
        
        # Initialize simulation
        U = Harris_sheet(B0, pert_amp)
        current_time = 0.0
        
        # Data logging
        data = []
        next_log_time = 0.0
        
        print(f"Running simulation: B0={B0}, eta={eta}, pert={pert_amp}, resolution={config.Nx}x{config.Ny}")
        
        while current_time < tmax:
            U, dt = RK4(U)
            current_time += dt
            
            # Log data at intervals
            if current_time >= next_log_time:
                # Calculate diagnostics
                E_total, E_kinetic, E_magnetic, E_internal = Total_Energy(U)
                divB = DivB(U)
                max_divB = np.max(np.abs(divB))
                rms_divB = np.sqrt(np.mean(divB**2))
                Ez_center = Electric_Field(U)
                
                # Current density metrics
                Jz = Jz_component(U)
                max_Jz = np.max(np.abs(Jz))
                mean_Jz = np.mean(np.abs(Jz))
                
                # Reconnection rate approximation
                rho, vx, vy, Bx, By, p = conserved_to_primitives(U)
                # Measure at inflow region
                inflow_region = (slice(config.Nx//4, 3*config.Nx//4), config.Ny//4)
                inflow_velocity = np.mean(np.abs(vy[inflow_region]))
                Alfven_velocity = np.mean(np.sqrt(Bx[inflow_region]**2 + By[inflow_region]**2) / 
                                        np.sqrt(rho[inflow_region]))
                reconnection_rate = inflow_velocity / Alfven_velocity if Alfven_velocity > 0 else 0
                
                log_entry = {
                    'time': current_time,
                    'E_total': E_total,
                    'E_kinetic': E_kinetic,
                    'E_magnetic': E_magnetic,
                    'E_internal': E_internal,
                    'max_divB': max_divB,
                    'rms_divB': rms_divB,
                    'Ez_center': Ez_center,
                    'max_Jz': max_Jz,
                    'mean_Jz': mean_Jz,
                    'reconnection_rate': reconnection_rate,
                    'B0': B0,
                    'eta': eta,
                    'pert_amp': pert_amp,
                    'Nx': config.Nx,
                    'Ny': config.Ny
                }
                
                data.append(log_entry)
                next_log_time += log_interval
                print(f"Time: {current_time:.3f}, Reconnection rate: {reconnection_rate:.4f}")
        
        # Restore original config
        if resolution:
            config.Nx, config.Ny = original_Nx, original_Ny
            config.dx = config.Lx / config.Nx
            config.dy = config.Ly / config.Ny
        config.ETA = original_eta
        
        # Save results
        df = pd.DataFrame(data)
        filename = f"{self.results_dir}/sim_B0_{B0}_eta_{eta}_pert_{pert_amp}_res_{config.Nx}x{config.Ny}_{datetime.now().strftime('%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved results to: {filename}")
        
        return df, filename
    
    def run_parameter_study(self, study_type='B0'):
        """Run systematic parameter studies"""
        
        base_params = {
            'B0': 0.5,
            'eta': 0.001, 
            'pert_amp': 0.1,
            'tmax': 1.0
        }
        
        if study_type == 'B0':
            values = [0.5, 0.6, 0.7, 0.8]
            param_name = 'B0'
        elif study_type == 'eta':
            values = [0.0001, 0.0005, 0.001, 0.005]
            param_name = 'eta'
        elif study_type == 'perturbation':
            values = [0.01, 0.05, 0.1, 0.2]
            param_name = 'pert_amp'
        elif study_type == 'resolution':
            values = [(64, 64), (128, 128), (256, 256)]
            param_name = 'resolution'
        else:
            raise ValueError("Unknown study type")
        
        results = []
        for value in values:
            params = base_params.copy()
            if param_name == 'resolution':
                df, filename = self.run_single_simulation(resolution=value, **{k: v for k, v in params.items() if k != param_name})
            else:
                params[param_name] = value
                df, filename = self.run_single_simulation(**params)
            
            # Calculate summary statistics
            summary = self.calculate_summary_statistics(df)
            summary['parameter_value'] = value
            summary['study_type'] = study_type
            results.append(summary)
        
        # Save study summary
        summary_df = pd.DataFrame(results)
        summary_filename = f"{self.results_dir}/summary_{study_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_filename, index=False)
        
        # Plot results
        self.plot_parameter_study(summary_df, study_type, param_name)
        
        return summary_df
    
    def calculate_summary_statistics(self, df):
        """Calculate key statistics from a simulation run"""
        return {
            'max_reconnection_rate': df['reconnection_rate'].max(),
            'final_reconnection_rate': df['reconnection_rate'].iloc[-1],
            'time_to_max_reconnection': df.loc[df['reconnection_rate'].idxmax(), 'time'],
            'max_Ez': df['Ez_center'].max(),
            'max_current_density': df['max_Jz'].max(),
            'energy_conservation_error': abs(df['E_total'].iloc[-1] - df['E_total'].iloc[0]) / df['E_total'].iloc[0],
            'magnetic_energy_loss': (df['E_magnetic'].iloc[0] - df['E_magnetic'].iloc[-1]) / df['E_magnetic'].iloc[0],
            'avg_divB_error': df['rms_divB'].mean(),
            'final_time': df['time'].iloc[-1]
        }
    
    def plot_parameter_study(self, summary_df, study_type, param_name):
        """Create plots for parameter study"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Parameter Study: {study_type}', fontsize=16)
        
        # Plot 1: Max reconnection rate vs parameter
        axes[0,0].plot(summary_df['parameter_value'], summary_df['max_reconnection_rate'], 'o-', linewidth=2)
        axes[0,0].set_xlabel(param_name)
        axes[0,0].set_ylabel('Max Reconnection Rate')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Magnetic energy loss vs parameter
        axes[0,1].plot(summary_df['parameter_value'], summary_df['magnetic_energy_loss'], 'o-', linewidth=2, color='red')
        axes[0,1].set_xlabel(param_name)
        axes[0,1].set_ylabel('Magnetic Energy Loss Fraction')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Energy conservation error vs parameter
        axes[1,0].plot(summary_df['parameter_value'], summary_df['energy_conservation_error'], 'o-', linewidth=2, color='green')
        axes[1,0].set_xlabel(param_name)
        axes[1,0].set_ylabel('Energy Conservation Error')
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Div(B) error vs parameter
        axes[1,1].plot(summary_df['parameter_value'], summary_df['avg_divB_error'], 'o-', linewidth=2, color='purple')
        axes[1,1].set_xlabel(param_name)
        axes[1,1].set_ylabel('Average Div(B) Error')
        axes[1,1].set_yscale('log')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f"{self.results_dir}/plots_{study_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved plot to: {plot_filename}")

def main():
    """Main function to run analyses"""
    analyzer = AnalysisRunner()
    
    print("MHD Simulation Analysis Runner")
    print("=" * 40)
    print("1. Run single simulation")
    print("2. Run B0 parameter study")
    print("3. Run resistivity (eta) study") 
    print("4. Run perturbation study")
    print("5. Run resolution study")
    print("6. Run all studies")
    
    choice = input("Enter your choice (1-6): ").strip()
    
    if choice == '1':
        # Single simulation with custom parameters
        B0 = float(input("B0 (default 1.0): ") or "1.0")
        eta = float(input("eta (default 0.01): ") or "0.01")
        pert_amp = float(input("perturbation (default 0.1): ") or "0.1")
        analyzer.run_single_simulation(B0=B0, eta=eta, pert_amp=pert_amp)
        
    elif choice == '2':
        analyzer.run_parameter_study('B0')
    elif choice == '3':
        analyzer.run_parameter_study('eta')
    elif choice == '4':
        analyzer.run_parameter_study('perturbation')
    elif choice == '5':
        analyzer.run_parameter_study('resolution')
    elif choice == '6':
        for study in ['B0', 'eta', 'perturbation', 'resolution']:
            print(f"\nRunning {study} study...")
            analyzer.run_parameter_study(study)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()