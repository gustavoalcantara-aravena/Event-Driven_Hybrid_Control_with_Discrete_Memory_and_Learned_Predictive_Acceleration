"""
Evaluation and Visualization: Aggregate results and generate figures/tables

Produces:
  - Tabla 1: Main comparison metrics
  - Tabla 2: Ablation study results
  - 5 figures: Architecture, tracking, compute, events, robustness
"""

import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent / "src")) # Removed

import numpy as np
import pandas as pd
import argparse
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import Logger

# Configure plot style
# sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-whitegrid') # Built-in matplotlib style or 'ggplot'
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


class EvaluationRunner:
    """Aggregate results and generate evaluation metrics"""
    
    def __init__(self,
                 results_dir: str = "results",
                 output_dir: str = "evaluation"):
        """
        Args:
            results_dir: directory with CSV results
            output_dir: directory for tables and figures
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger(name="evaluate", log_dir="logs")
    
    def load_all_results(self, plant_type: str) -> pd.DataFrame:
        """
        Load all result files for a plant
        
        Args:
            plant_type: "motor" or "oven"
        
        Returns:
            combined DataFrame
        """
        # Find plant-specific files
        files = list(self.results_dir.glob(f"results_{plant_type}_*.csv"))
        
        # Check for both-combined file
        combined_file = self.results_dir / "results_both_combined.csv"
        
        dfs = []
        # Load plant-specific files
        for f in files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
                print(f"  Loaded {f.name}: {len(df)} rows")
            except Exception as e:
                print(f"  Warning: Could not load {f}: {e}")
        
        # Load and filter from the combined file if it exists
        if combined_file.exists():
            try:
                df_comb = pd.read_csv(combined_file)
                # Filter by plant column
                if 'plant' in df_comb.columns:
                    df_plant = df_comb[df_comb['plant'] == plant_type]
                elif 'plant_name' in df_comb.columns:
                    df_plant = df_comb[df_comb['plant_name'] == plant_type]
                else:
                    df_plant = df_comb
                
                if not df_plant.empty:
                    dfs.append(df_plant)
                    print(f"  Loaded {len(df_plant)} rows for {plant_type} from results_both_combined.csv")
            except Exception as e:
                print(f"  Warning: Could not load {combined_file.name}: {e}")
        
        if not dfs:
            raise ValueError(f"No result files found for {plant_type}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        # Drop duplicates if any
        combined_df = combined_df.drop_duplicates(subset=['seed', 'scenario', 'method'])
        
        return combined_df
    
    def compute_table1_main_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tabla 1: Main comparison metrics (Proposed + 3 Baselines)
        
        Metrics:
          - Cost: total_cost (lower better)
          - Tracking Error: tracking_error_mse
          - Violations: num_violations
          - Compute Time: cpu_time_mean
          - Event Rate: event_rate
        """
        # Methods: Proposed, B1_PeriodicMPC, B2_ClassicEMPC, B3_RLnoMemory
        methods = ['Proposed', 'B1_PeriodicMPC', 'B2_ClassicEMPC', 'B3_RLnoMemory']
        
        results = []
        
        for method in methods:
            df_method = df[df['method'] == method]
            
            if len(df_method) == 0:
                continue
            
            row = {
                'Method': method,
                'Cost (↓)': f"{df_method['total_cost'].mean():.4f} ± {df_method['total_cost'].std():.4f}",
                'Tracking MSE (↓)': f"{df_method['tracking_error_mse'].mean():.4f} ± {df_method['tracking_error_mse'].std():.4f}",
                'Violations (↓)': f"{df_method['num_violations'].mean():.1f} ± {df_method['num_violations'].std():.1f}",
                'Compute [ms] (↓)': f"{df_method['cpu_time_mean'].mean()*1000:.2f} ± {df_method['cpu_time_mean'].std()*1000:.2f}",
                'Event Rate (↓)': f"{df_method['event_rate'].mean():.3f} ± {df_method['event_rate'].std():.3f}",
            }
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def compute_table2_ablations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tabla 2: Ablation study results
        
        Variants: Proposed, A1_NoMemory, A2_NoLAPA, A3_Periodic, A4_EventMPC
        """
        ablations = ['Proposed', 'A1_NoMemory', 'A2_NoLAPA', 'A3_Periodic', 'A4_EventMPC']
        
        results = []
        
        for ablation in ablations:
            df_ab = df[df['method'] == ablation]
            
            if len(df_ab) == 0:
                continue
            
            row = {
                'Variant': ablation,
                'Cost': f"{df_ab['total_cost'].mean():.4f}",
                'Track Error': f"{df_ab['tracking_error_mse'].mean():.4f}",
                'Violations': f"{df_ab['num_violations'].mean():.1f}",
                'CPU Time [ms]': f"{df_ab['cpu_time_mean'].mean()*1000:.2f}",
                'Events': f"{df_ab['num_events'].mean():.0f}",
            }
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def plot_architecture_diagram(self):
        """
        Figura 1: System Architecture Diagram (text-based)
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Event-Driven Hybrid Control Architecture',
               ha='center', fontsize=16, fontweight='bold',
               transform=ax.transAxes)
        
        # Main components (boxes)
        components = [
            (0.1, 0.8, 'Plant\n(Motor/Oven)', 'blue'),
            (0.3, 0.8, 'State Sensor\n(x_k)', 'green'),
            (0.5, 0.8, 'LSTM Predictor\n(ŷ_{k|k-1})', 'orange'),
            (0.7, 0.8, 'Event Trigger\nE(x, ŷ, m)', 'red'),
            (0.9, 0.8, 'Discrete Memory\n(3-bit flip-flop)', 'purple'),
            
            (0.3, 0.5, 'MPC Solver\n(CasADi/IPOPT)', 'darkblue'),
            (0.7, 0.5, 'LAPA acceleration\nA/B', 'darkgreen'),
            
            (0.5, 0.2, 'Control u_k', 'black'),
        ]
        
        for x, y, label, color in components:
            bbox = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3)
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=11, bbox=bbox, transform=ax.transAxes)
        
        # Arrows (connections)
        arrows = [
            ((0.15, 0.8), (0.28, 0.8)),  # Plant → Sensor
            ((0.35, 0.8), (0.48, 0.8)),  # Sensor → LSTM
            ((0.55, 0.8), (0.68, 0.8)),  # LSTM → Trigger
            ((0.75, 0.8), (0.88, 0.8)),  # Trigger → Memory
            
            ((0.5, 0.75), (0.35, 0.55)),  # Trigger → MPC
            ((0.75, 0.75), (0.65, 0.55)),  # Memory → LAPA
            
            ((0.4, 0.45), (0.5, 0.25)),  # MPC → Control
            ((0.6, 0.45), (0.5, 0.25)),  # LAPA → Control
            ((0.5, 0.15), (0.15, 0.8)),  # Control → Plant
        ]
        
        for (x1, y1), (x2, y2) in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2),
                       transform=ax.transAxes)
        
        # Legend
        legend_text = (
            "Algorithm 1 Loop:\n"
            "1. Read state x_k, get reference\n"
            "2. LSTM prediction ŷ_{k|k-1}\n"
            "3. Evaluate trigger E(x, ŷ, m) vs η(m)\n"
            "4. Update memory m_k → m_{k+1}\n"
            "5. IF trigger: MPC with LAPA\n"
            "6. Apply control, log metrics"
        )
        
        ax.text(0.05, 0.35, legend_text,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               transform=ax.transAxes, family='monospace')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig1_Architecture.png', dpi=150, bbox_inches='tight')
        print(f"✓ Fig1_Architecture.png")
    
    def plot_tracking_comparison(self, df: pd.DataFrame):
        """
        Figura 2: Tracking Performance Comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Tracking Performance Comparison (Motor & Oven)', fontsize=14, fontweight='bold')
        
        methods = df['method'].unique()
        methods = sorted([m for m in methods if not m.startswith('A')])  # Exclude ablations
        
        # Cost comparison
        ax = axes[0, 0]
        data = [df[df['method'] == m]['total_cost'].values for m in methods]
        bp = ax.boxplot(data, labels=[m.replace('B1_', '').replace('B2_', '').replace('B3_', '') 
                                      for m in methods])
        ax.set_ylabel('Total Cost')
        ax.set_title('Cost per Episode')
        ax.grid(True, alpha=0.3)
        
        # Tracking error
        ax = axes[0, 1]
        data = [df[df['method'] == m]['tracking_error_mse'].values for m in methods]
        ax.boxplot(data, labels=[m.replace('B1_', '').replace('B2_', '').replace('B3_', '')
                                 for m in methods])
        ax.set_ylabel('MSE')
        ax.set_title('Tracking Error (MSE)')
        ax.grid(True, alpha=0.3)
        
        # Violations
        ax = axes[1, 0]
        data = [df[df['method'] == m]['num_violations'].values for m in methods]
        ax.boxplot(data, labels=[m.replace('B1_', '').replace('B2_', '').replace('B3_', '')
                                 for m in methods])
        ax.set_ylabel('Count')
        ax.set_title('Constraint Violations')
        ax.grid(True, alpha=0.3)
        
        # MAE
        ax = axes[1, 1]
        data = [df[df['method'] == m]['tracking_error_mae'].values for m in methods]
        ax.boxplot(data, labels=[m.replace('B1_', '').replace('B2_', '').replace('B3_', '')
                                 for m in methods])
        ax.set_ylabel('MAE')
        ax.set_title('Tracking Error (MAE)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig2_Tracking.png', dpi=150, bbox_inches='tight')
        print(f"✓ Fig2_Tracking.png")
    
    def plot_computational_efficiency(self, df: pd.DataFrame):
        """
        Figura 3: Computational Cost Analysis
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Computational Efficiency', fontsize=14, fontweight='bold')
        
        methods = df['method'].unique()
        
        # CPU time
        ax = axes[0]
        cpu_times = df.groupby('method')['cpu_time_mean'].agg(['mean', 'std'])
        cpu_times.plot(kind='bar', y='mean', xerr='std', ax=ax, legend=False)
        ax.set_ylabel('CPU Time [seconds]')
        ax.set_title('Average Compute Time per Episode')
        ax.set_xlabel('Method')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Time percentiles
        ax = axes[1]
        p95_times = df.groupby('method')['cpu_time_p95'].mean()
        mean_times = df.groupby('method')['cpu_time_mean'].mean()
        
        x = np.arange(len(p95_times))
        width = 0.35
        
        ax.bar(x - width/2, mean_times, width, label='Mean', alpha=0.8)
        ax.bar(x + width/2, p95_times, width, label='P95', alpha=0.8)
        
        ax.set_ylabel('CPU Time [seconds]')
        ax.set_title('Mean vs 95th Percentile')
        ax.set_xticks(x)
        ax.set_xticklabels(p95_times.index, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig3_Compute.png', dpi=150, bbox_inches='tight')
        print(f"✓ Fig3_Compute.png")
    
    def plot_event_statistics(self, df: pd.DataFrame):
        """
        Figura 4: Event Triggering Statistics
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Event-Triggered Control Statistics', fontsize=14, fontweight='bold')
        
        # Event rate
        ax = axes[0]
        event_rates = df.groupby('method')['event_rate'].mean().sort_values()
        event_rates.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Event Rate (events per step)')
        ax.set_title('Average Event Trigger Rate')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Number of events per episode
        ax = axes[1]
        num_events = df.groupby('method')['num_events'].agg(['mean', 'std'])
        num_events.plot(kind='bar', y='mean', xerr='std', ax=ax, legend=False)
        ax.set_ylabel('Count')
        ax.set_title('Events per Episode')
        ax.set_xlabel('Method')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Inter-event time distribution (estimated from event rate)
        ax = axes[2]
        # Avoid division by zero
        mean_rates = df.groupby('method')['event_rate'].mean()
        inter_event = 1.0 / (mean_rates + 1e-6)
        inter_event.plot(kind='bar', ax=ax, color='coral')
        ax.set_ylabel('Mean Inter-Event Time [steps]')
        ax.set_title('Inter-Event Statistics')
        ax.set_xlabel('Method')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig4_Events.png', dpi=150, bbox_inches='tight')
        print(f"✓ Fig4_Events.png")
    
    def plot_robustness_analysis(self, df: pd.DataFrame):
        """
        Figura 5: Robustness Analysis (variability across seeds/scenarios)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Robustness Analysis (Variability Across Experiments)', fontsize=14, fontweight='bold')
        
        proposed_df = df[df['method'] == 'Proposed']
        
        # Cost std by scenario
        ax = axes[0, 0]
        cost_by_scenario = proposed_df.groupby('scenario')['total_cost'].std()
        ax.plot(cost_by_scenario.index, cost_by_scenario.values, marker='o', linewidth=2)
        ax.set_xlabel('Scenario Index')
        ax.set_ylabel('Cost Standard Deviation')
        ax.set_title('Cost Variability Across Seeds')
        ax.grid(True, alpha=0.3)
        
        # Tracking error consistency
        ax = axes[0, 1]
        track_by_seed = proposed_df.groupby('seed')['tracking_error_mse'].mean()
        ax.bar(track_by_seed.index, track_by_seed.values, alpha=0.7)
        ax.set_xlabel('Seed')
        ax.set_ylabel('Mean Tracking Error')
        ax.set_title('Tracking Error Consistency')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Violations by scenario
        ax = axes[1, 0]
        violations_by_scenario = proposed_df.groupby('scenario')['num_violations'].mean()
        ax.plot(violations_by_scenario.index, violations_by_scenario.values, marker='s', color='red', linewidth=2)
        ax.set_xlabel('Scenario Index')
        ax.set_ylabel('Avg Violations')
        ax.set_title('Violations Across Scenarios')
        ax.grid(True, alpha=0.3)
        
        # Compute time variability
        ax = axes[1, 1]
        cpu_cv = (proposed_df.groupby('seed')['cpu_time_mean'].std() / 
                 proposed_df.groupby('seed')['cpu_time_mean'].mean())
        ax.bar(cpu_cv.index, cpu_cv.values, alpha=0.7, color='green')
        ax.set_xlabel('Seed')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Compute Time Consistency (CV)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Fig5_Robustness.png', dpi=150, bbox_inches='tight')        
        self.logger.info(f"✓ Fig5_Robustness.png")
    
    def plot_trajectory_comparison(self, df: pd.DataFrame):
        """
        Figura 6: Temporal Trajectory Comparison (Real Experimental Data)
        Shows how controllers track reference over time with event markers
        """
        import json
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        for idx, plant in enumerate(['motor', 'oven']):
            # Select methods to plot
            methods_to_plot = ['Proposed', 'B1_PeriodicMPC', 'B3_RLnoMemory']
            colors = {'Proposed': '#1f77b4', 'B1_PeriodicMPC': '#ff7f0e', 'B3_RLnoMemory': '#2ca02c'}
            
            # Subplot 1: State trajectory (position/temperature)
            ax1 = axes[idx, 0]
            # Subplot 2: Control input
            ax2 = axes[idx, 1]
            
            # Labels
            if plant == 'motor':
                ylabel_state = 'Position (rad)'
                ylabel_control = 'Voltage (V)'
            else:
                ylabel_state = 'Temperature (°C)'
                ylabel_control = 'Power (%)'
            
            # Load and plot real trajectories
            for method in methods_to_plot:
                traj_file = Path("trajectories") / plant / f"{method}.json"
                
                if not traj_file.exists():
                    print(f"Warning: Trajectory file not found: {traj_file}")
                    continue
                
                # Load trajectory data
                with open(traj_file, 'r') as f:
                    traj_data = json.load(f)
                
                # Extract data
                states = np.array(traj_data['states'])  # Shape: (num_steps, state_dim)
                controls = np.array(traj_data['controls'])
                references = np.array(traj_data['references'])
                events = np.array(traj_data['events'])
                num_steps = traj_data['num_steps']
                
                time = np.arange(num_steps)
                
                # Plot state (first dimension: position or temperature)
                state_0 = states[:, 0]
                ax1.plot(time, state_0, linewidth=2, label=method, color=colors.get(method, 'gray'), alpha=0.8)
                
                # Mark events
                event_times = time[events]
                if len(event_times) > 0:
                    ax1.scatter(event_times, state_0[events], 
                               c=colors.get(method, 'gray'), s=60, marker='o', 
                               edgecolors='black', linewidth=1.5, zorder=5)
                
                # Plot control
                ax2.plot(time, controls, linewidth=2, label=method, color=colors.get(method, 'gray'), alpha=0.8)
            
            # Plot reference (use first method's reference)
            first_method = methods_to_plot[0]
            ref_file = Path("trajectories") / plant / f"{first_method}.json"
            if ref_file.exists():
                with open(ref_file, 'r') as f:
                    ref_data = json.load(f)
                references = np.array(ref_data['references'])
                time = np.arange(len(references))
                ax1.plot(time, references, 'k--', linewidth=2.5, label='Reference', alpha=0.7, zorder=1)
            
            # Customize state plot
            ax1.set_title(f'{plant.upper()}: State Trajectory', fontsize=13, fontweight='bold')
            ax1.set_xlabel('Time Step', fontsize=11)
            ax1.set_ylabel(ylabel_state, fontsize=11)
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Customize control plot
            ax2.set_title(f'{plant.upper()}: Control Input', fontsize=13, fontweight='bold')
            ax2.set_xlabel('Time Step', fontsize=11)
            ax2.set_ylabel(ylabel_control, fontsize=11)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "Fig6_Trajectories.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Fig6_Trajectories.png (Real experimental trajectories)")
    
    def plot_pareto_front(self, df: pd.DataFrame):
        """
        Figura 7: Pareto Front (Cost vs Event Rate)
        Demonstrates trade-off between performance and communication
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, plant in enumerate(['motor', 'oven']):
            ax = axes[idx]
            df_plant = df[df['plant'] == plant]
            
            # Aggregate by method
            methods = df_plant['method'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
            
            for i, method in enumerate(methods):
                df_method = df_plant[df_plant['method'] == method]
                
                # Mean values
                mean_cost = df_method['total_cost'].mean()
                mean_event_rate = df_method['event_rate'].mean()
                mean_cpu = df_method['cpu_time_mean'].mean() * 1000  # Convert to ms
                
                # Plot with size proportional to CPU time
                ax.scatter(mean_event_rate, mean_cost, 
                          s=mean_cpu*50 + 100,  # Scale for visibility
                          c=[colors[i]], 
                          alpha=0.7, 
                          edgecolors='black',
                          linewidth=1.5,
                          label=method)
                
                # Add method label
                ax.annotate(method.replace('_', '\n'), 
                           (mean_event_rate, mean_cost),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
            
            ax.set_xlabel('Event Rate (events/step)', fontsize=12)
            ax.set_ylabel('Total Cost (↓)', fontsize=12)
            ax.set_title(f'{plant.upper()}: Cost vs Communication Trade-off', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            
            # Add note about bubble size
            ax.text(0.02, 0.98, 'Bubble size ∝ CPU time', 
                   transform=ax.transAxes, fontsize=9, 
                   verticalalignment='top', alpha=0.7)
        
        plt.tight_layout()
        output_path = self.output_dir / "Fig7_ParetoFront.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Fig7_ParetoFront.png")
    
    def compute_statistical_tests(self, df: pd.DataFrame):
        """
        Table 3: Statistical Significance Tests
        Wilcoxon signed-rank tests comparing Proposed vs Baselines
        """
        from scipy.stats import wilcoxon, mannwhitneyu
        
        results = []
        
        for plant in ['motor', 'oven']:
            df_plant = df[df['plant'] == plant]
            
            # Get Proposed method data
            proposed_data = df_plant[df_plant['method'] == 'Proposed']
            
            # Baselines to compare
            baselines = ['B1_PeriodicMPC', 'B2_ClassicEMPC', 'B3_RLnoMemory']
            metrics = ['total_cost', 'tracking_error_mse', 'num_violations', 'cpu_time_mean', 'event_rate']
            
            for baseline in baselines:
                baseline_data = df_plant[df_plant['method'] == baseline]
                
                if len(baseline_data) == 0:
                    continue
                
                row = {'Plant': plant, 'Comparison': f'Proposed vs {baseline}'}
                
                for metric in metrics:
                    try:
                        # Use Mann-Whitney U test (non-parametric, independent samples)
                        stat, p_value = mannwhitneyu(
                            proposed_data[metric].dropna(),
                            baseline_data[metric].dropna(),
                            alternative='two-sided'
                        )
                        
                        # Significance markers
                        if p_value < 0.001:
                            sig = '***'
                        elif p_value < 0.01:
                            sig = '**'
                        elif p_value < 0.05:
                            sig = '*'
                        else:
                            sig = 'ns'
                        
                        row[metric] = f"p={p_value:.4f} {sig}"
                    except Exception as e:
                        row[metric] = 'N/A'
                
                results.append(row)
        
        # Create DataFrame
        stats_df = pd.DataFrame(results)
        
        # Save to CSV
        output_path = self.output_dir / "Table3_StatisticalTests.csv"
        stats_df.to_csv(output_path, index=False)
        
        print("\n✓ Table3_StatisticalTests.csv")
        print(stats_df.to_string(index=False))
        print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
        self.logger.info(f"✓ Table3_StatisticalTests.csv")
    
    def plot_radar_chart(self, df: pd.DataFrame):
        """
        Figura 10: Radar Chart (Normalized Multi-dimensional Comparison)
        """
        from math import pi
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(projection='polar'))
        
        # Metrics to compare (normalized 0-1, higher is better)
        metrics = ['Cost', 'Tracking', 'Violations', 'CPU', 'Events']
        num_vars = len(metrics)
        
        for idx, plant in enumerate(['motor', 'oven']):
            ax = axes[idx]
            df_plant = df[df['plant'] == plant]
            
            # Methods to compare
            methods_to_plot = ['Proposed', 'B1_PeriodicMPC', 'B2_ClassicEMPC', 'B3_RLnoMemory']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            # Compute angles for radar chart
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]  # Complete the circle
            
            for i, method in enumerate(methods_to_plot):
                df_method = df_plant[df_plant['method'] == method]
                
                if len(df_method) == 0:
                    continue
                
                # Normalize metrics (invert so higher is better)
                values = [
                    1 - (df_method['total_cost'].mean() / df_plant['total_cost'].max()),  # Lower cost is better
                    1 - (df_method['tracking_error_mse'].mean() / df_plant['tracking_error_mse'].max()),
                    1 - (df_method['num_violations'].mean() / max(df_plant['num_violations'].max(), 1)),
                    1 - (df_method['cpu_time_mean'].mean() / max(df_plant['cpu_time_mean'].max(), 1e-6)),
                    1 - (df_method['event_rate'].mean() / max(df_plant['event_rate'].max(), 1))
                ]
                values += values[:1]  # Complete the circle
                
                # Plot
                ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
                ax.fill(angles, values, alpha=0.15, color=colors[i])
            
            # Customize
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            ax.set_title(f'{plant.upper()}: Multi-dimensional Performance', 
                        fontsize=13, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "Fig10_RadarChart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Fig10_RadarChart.png")
    
    def compute_computational_budget(self, df: pd.DataFrame):
        """
        Table 4: Computational Budget Analysis
        Real-time feasibility assessment
        """
        results = []
        
        # Real-time constraints
        rt_limits = {
            'motor': 10.0,  # 10ms for motor (100Hz control)
            'oven': 100.0   # 100ms for oven (10Hz control)
        }
        
        for plant in ['motor', 'oven']:
            df_plant = df[df['plant'] == plant]
            
            for method in df_plant['method'].unique():
                df_method = df_plant[df_plant['method'] == method]
                
                # Convert to milliseconds
                cpu_mean = df_method['cpu_time_mean'].mean() * 1000
                cpu_std = df_method['cpu_time_std'].mean() * 1000
                cpu_p95 = df_method['cpu_time_p95'].mean() * 1000
                cpu_max = df_method['cpu_time_p95'].max() * 1000
                
                # Real-time feasibility
                rt_limit = rt_limits[plant]
                feasible = "✓ Yes" if cpu_p95 < rt_limit else "✗ No"
                margin = ((rt_limit - cpu_p95) / rt_limit) * 100
                
                results.append({
                    'Plant': plant,
                    'Method': method,
                    'Mean CPU (ms)': f"{cpu_mean:.4f}",
                    'Std CPU (ms)': f"{cpu_std:.4f}",
                    'P95 CPU (ms)': f"{cpu_p95:.4f}",
                    'Max CPU (ms)': f"{cpu_max:.4f}",
                    'RT Limit (ms)': f"{rt_limit:.1f}",
                    'RT Feasible': feasible,
                    'Safety Margin (%)': f"{margin:.1f}%"
                })
        
        budget_df = pd.DataFrame(results)
        
        # Save to CSV
        output_path = self.output_dir / "Table4_ComputationalBudget.csv"
        budget_df.to_csv(output_path, index=False)
        
        print("\n✓ Table4_ComputationalBudget.csv")
        print(budget_df.to_string(index=False))
        
        self.logger.info(f"✓ Table4_ComputationalBudget.csv")
    
    def run_evaluation(self, plants: List[str] = ['motor', 'oven']):
        """
        Run complete evaluation pipeline
        
        Args:
            plants: list of plant types
        """
        all_dfs = []
        for plant in plants:
            print(f"\n{'='*70}")
            print(f"EVALUATING {plant.upper()}")
            print(f"{'='*70}")
            
            # Load results
            df = self.load_all_results(plant)
            all_dfs.append(df)
            self.logger.info(f"Loaded {len(df)} total results for {plant}")
            
            # Generate tables
            print("\nGenerating Tables...")
            
            table1 = self.compute_table1_main_metrics(df)
            table1_file = self.output_dir / f"Table1_MainMetrics_{plant}.csv"
            table1.to_csv(table1_file, index=False)
            print(f"✓ Table1_MainMetrics_{plant}.csv")
            print(table1.to_string())
            
            table2 = self.compute_table2_ablations(df)
            table2_file = self.output_dir / f"Table2_Ablations_{plant}.csv"
            table2.to_csv(table2_file, index=False)
            print(f"\n✓ Table2_Ablations_{plant}.csv")
            print(table2.to_string())
            
            # Generate per-plant figures
            print("\nGenerating Figures...")
            
            self.plot_architecture_diagram()
            self.plot_tracking_comparison(df)
            self.plot_computational_efficiency(df)
            self.plot_event_statistics(df)
            self.plot_robustness_analysis(df)
            
            # NEW: Additional tables
            print("\nGenerating Additional Tables...")
            self.compute_statistical_tests(df)  # Table3
            self.compute_computational_budget(df)  # Table4
            
        # Combined Visualizations
        if all_dfs:
            combined_all = pd.concat(all_dfs, ignore_index=True)
            print("\nGenerating Enhanced Combined Visualizations...")
            self.plot_trajectory_comparison(combined_all)  # Fig6
            self.plot_pareto_front(combined_all)  # Fig7
            self.plot_radar_chart(combined_all)  # Fig10
        
        print(f"\n{'='*70}")
        print("✓ Evaluation complete!")
        print(f"{'='*70}")
        print(f"Results saved to: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize results")
    
    parser.add_argument('--results', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--output', type=str, default='evaluation',
                       help='Output directory')
    parser.add_argument('--plants', type=str, default='motor,oven',
                       help='Plants to evaluate (comma-separated)')
    
    args = parser.parse_args()
    
    plants = [p.strip() for p in args.plants.split(',')]
    
    runner = EvaluationRunner(
        results_dir=args.results,
        output_dir=args.output
    )
    
    runner.run_evaluation(plants=plants)


if __name__ == "__main__":
    main()
