#!/usr/bin/env python3
"""
Create performance plots from the benchmark results.
"""

import json
import sys
from pathlib import Path

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("❌ Matplotlib/seaborn not available")
    print("   On macOS: brew install python-tk")
    print("   Or: pip install matplotlib seaborn")
    sys.exit(1)


def create_plots_from_reports(reports_dir):
    """Create plots from benchmark reports."""
    reports_path = Path(reports_dir)
    
    if not reports_path.exists():
        print(f"❌ Reports directory not found: {reports_path}")
        return
    
    # Load summary data
    summary_file = reports_path / "summary.json"
    if not summary_file.exists():
        print(f"❌ Summary file not found: {summary_file}")
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Extract successful results
    successful_results = [r for r in summary['results'] if r['success']]
    
    if not successful_results:
        print("❌ No successful results to plot")
        return
    
    # Group by implementation
    by_impl = {}
    for result in successful_results:
        impl = result['implementation']
        if impl not in by_impl:
            by_impl[impl] = {'scales': [], 'times': [], 'ops_per_sec': []}
        by_impl[impl]['scales'].append(result['scale'])
        by_impl[impl]['times'].append(result['execution_time_sec'])
        by_impl[impl]['ops_per_sec'].append(result['operations_per_sec'])
    
    # Sort scales for each implementation
    for impl, data in by_impl.items():
        # Sort by scale
        sorted_data = sorted(zip(data['scales'], data['times'], data['ops_per_sec']))
        data['scales'] = [x[0] for x in sorted_data]
        data['times'] = [x[1] for x in sorted_data]
        data['ops_per_sec'] = [x[2] for x in sorted_data]
    
    print(f"📊 Creating plots for {len(by_impl)} implementations...")
    
    # Create execution time plot (matching original customer plot style)
    plt.figure(figsize=(10, 6))
    
    # Use consistent colors and styling
    colors = {'memory': 'orange', 'redis': 'darkgreen', 'mongodb': 'blue', 'sqlite': 'red', 'postgres': 'purple'}
    
    for impl, data in by_impl.items():
        color = colors.get(impl, 'gray')
        label = impl.capitalize().replace('_', '-')  # Format like original
        plt.plot(data['scales'], data['times'], 'o-', 
                label=label, color=color, linewidth=2, markersize=6)
    
    plt.xlabel('Parallel graph async executions', fontsize=12)
    plt.ylabel('Time (s)', fontsize=12)
    plt.title('Graph execution time by checkpointers', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save execution time plot
    exec_plot_path = reports_path / "execution_time_by_checkpointers.png"
    plt.savefig(exec_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📈 {exec_plot_path.name}")
    
    # Create throughput plot
    plt.figure(figsize=(10, 6))
    
    for impl, data in by_impl.items():
        color = colors.get(impl, 'gray')
        label = impl.capitalize().replace('_', '-')
        plt.plot(data['scales'], data['ops_per_sec'], 'o-',
                label=label, color=color, linewidth=2, markersize=6)
    
    plt.xlabel('Parallel graph async executions', fontsize=12)
    plt.ylabel('Operations per Second', fontsize=12)
    plt.title('Throughput by checkpointers', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save throughput plot
    throughput_plot_path = reports_path / "throughput_by_checkpointers.png"
    plt.savefig(throughput_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📈 {throughput_plot_path.name}")
    
    # Print summary of what we plotted
    print(f"\n✅ Plotted data for:")
    for impl, data in by_impl.items():
        scales_range = f"{min(data['scales'])}-{max(data['scales'])}"
        print(f"   - {impl}: {len(data['scales'])} points (scales {scales_range})")


if __name__ == "__main__":
    # Find the latest report directory
    reports_base = Path("reports")
    
    if len(sys.argv) > 1:
        # Use specified directory
        reports_dir = sys.argv[1]
    else:
        # Find latest run
        if not reports_base.exists():
            print(f"❌ Reports base directory not found: {reports_base}")
            sys.exit(1)
        
        run_dirs = list(reports_base.glob("run-*"))
        if not run_dirs:
            print(f"❌ No run directories found in {reports_base}")
            sys.exit(1)
        
        latest_dir = max(run_dirs, key=lambda d: d.name)
        reports_dir = str(latest_dir)
    
    print(f"📊 Creating plots from: {reports_dir}")
    create_plots_from_reports(reports_dir)