"""
Lux Scientia - Main Entry Point

Phase 1: Digital Bedrock
- Visualize the rover moving in MuJoCo
- Collect noisy sensor data
- Verify the simulation works correctly
"""

# Standard library imports
import argparse
import time
from typing import Optional

# Third-party imports
import numpy as np

# Local imports
from src.environment.puffer_wrapper import LuxEnvironment


def phase1_visualization(render: bool = True, steps: int = 100):
    """Phase 1: Test environment with visualization.
    
    Args:
        render: Show PyBullet GUI
        steps: Number of simulation steps to run
    """
    print("=" * 60)
    print("PHASE 1: Digital Bedrock - Environment Validation")
    print("=" * 60)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    render_mode = "human" if render else None
    env = LuxEnvironment(render_mode=render_mode, max_ticks=500)
    print(f"✓ MuJoCo environment created (render={'ON' if render else 'OFF'})")
    
    # Reset environment
    print("\n[2/4] Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"✓ Initial observation: x={obs[0]:.2f}m, y={obs[1]:.2f}m, lux={obs[2]:.2f}")
    
    # Run random walk
    print(f"\n[3/4] Running {steps} steps with random actions...")
    observations = []
    
    for step in range(steps):
        # Random action in [-1, 1]
        action = np.random.uniform(-1, 1, size=2).astype(np.float32)
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        observations.append(obs)
        
        # Print progress
        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{steps}: x={obs[0]:.2f}m, y={obs[1]:.2f}m, lux={obs[2]:.2f}")
        
        # Small delay if rendering
        if render:
            time.sleep(0.01)
        
        # Check autoreset
        if done or truncated:
            print(f"  Episode ended at step {step + 1}, autoreset triggered")
    
    # Analyze collected data
    print("\n[4/4] Analyzing collected data...")
    observations = np.array(observations)
    
    print(f"✓ Collected {len(observations)} observations")
    print(f"  Position range: x=[{observations[:,0].min():.2f}, {observations[:,0].max():.2f}]m")
    print(f"  Position range: y=[{observations[:,1].min():.2f}, {observations[:,1].max():.2f}]m")
    print(f"  Lux range: [{observations[:,2].min():.2f}, {observations[:,2].max():.2f}]")
    print(f"  Mean lux: {observations[:,2].mean():.2f} ± {observations[:,2].std():.2f}")
    
    # Cleanup
    env.close()
    
    print("\n" + "=" * 60)
    print("Phase 1 Complete! Environment is working correctly.")
    print("=" * 60)
    
    return observations


def phase2_data_collection(steps: int = 500, save_csv: bool = False):
    """Phase 2 Preview: Collect data for offline analysis.
    
    Args:
        steps: Number of steps to collect
        save_csv: Whether to save data to CSV
    """
    print("=" * 60)
    print("PHASE 2 PREVIEW: Data Collection for Offline Analysis")
    print("=" * 60)
    
    env = LuxEnvironment(render_mode=None, max_ticks=5000)
    obs, _ = env.reset(seed=42)
    
    data = []
    print(f"\nCollecting {steps} data points...")
    
    for step in range(steps):
        action = np.random.uniform(-1, 1, size=2).astype(np.float32)
        obs, reward, done, truncated, info = env.step(action)
        
        # Store [x, y, lux]
        data.append(obs)
        
        if (step + 1) % 100 == 0:
            print(f"  Collected {step + 1}/{steps} samples...")
    
    data = np.array(data)
    env.close()
    
    if save_csv:
        import csv
        filename = "lux_data.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'lux'])
            writer.writerows(data)
        print(f"\n✓ Data saved to {filename}")
        print("  You can now use this data for offline PySR testing!")
    
    print(f"\n✓ Collected {len(data)} samples")
    print(f"  Position coverage: x=[{data[:,0].min():.2f}, {data[:,0].max():.2f}]m")
    print(f"  Position coverage: y=[{data[:,1].min():.2f}, {data[:,1].max():.2f}]m")
    print(f"  Lux range: [{data[:,2].min():.2f}, {data[:,2].max():.2f}]")
    
    return data


def main():
    """Main entry point with phase selection."""
    parser = argparse.ArgumentParser(
        description="Lux Scientia - Autonomous Robot Scientist"
    )
    parser.add_argument(
        '--phase',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='Execution phase: 1=Visualization, 2=Data Collection, 3=Active Inference (future)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Show PyBullet GUI visualization'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--save-csv',
        action='store_true',
        help='Save collected data to CSV (Phase 2)'
    )
    
    args = parser.parse_args()
    
    if args.phase == 1:
        phase1_visualization(render=args.render, steps=args.steps)
    elif args.phase == 2:
        phase2_data_collection(steps=args.steps, save_csv=args.save_csv)
    elif args.phase == 3:
        print("Phase 3 (Active Inference) - Coming soon!")
        print("This will include:")
        print("  - PySR symbolic regression")
        print("  - Curiosity-driven exploration")
        print("  - LLM discovery validation")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
