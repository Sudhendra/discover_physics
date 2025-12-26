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
import yaml
from pathlib import Path
from typing import Optional

# Third-party imports
import numpy as np

# Local imports
from src.environment.puffer_wrapper import LuxEnvironment
from src.agent.perception import PerceptionBuffer
from src.agent.theorist import Theorist
from src.agent.navigator import Navigator


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


def phase2_scientist_loop(steps: int = 1000, update_freq: int = 100, render: bool = False, 
                          mode: str = "active"):
    """Phase 2: The Loop - Integrate Scientist Engine with Active Learning.
    
    Connects the perception buffer with PySR symbolic regression.
    Uses active learning to explore efficiently.
    
    Args:
        steps: Number of simulation steps
        update_freq: Steps between PySR regression updates
        render: Show MuJoCo visualization
        mode: Navigation mode - 'random', 'active', or 'curious'
    """
    print("=" * 60)
    print("PHASE 2: The Loop - Scientist Engine Integration")
    print("=" * 60)
    
    # Load configuration
    config_path = Path("config/agent_params.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Loaded configuration from config/agent_params.yaml")
    else:
        config = {}
        print("⚠  No config file, using defaults")
    
    # Initialize components
    print(f"\n[1/5] Initializing Scientist Engine (mode={mode})...")
    render_mode = "human" if render else None
    env = LuxEnvironment(render_mode=render_mode, max_ticks=10000)
    buffer = PerceptionBuffer(
        max_size=config.get('perception', {}).get('max_buffer_size', 2000),
        grid_size=10  # 10x10 coverage grid
    )
    theorist = Theorist(config=config.get('pysr', {}))
    navigator = Navigator(
        mode=mode,
        surprise_threshold=config.get('curiosity', {}).get('surprise_threshold', 0.15)
    )
    
    print(f"  ✓ Environment: MuJoCo (render={'ON' if render else 'OFF'})")
    print(f"  ✓ Buffer: max_size={buffer.max_size}, grid={buffer.grid_size}x{buffer.grid_size}")
    print(f"  ✓ Theorist: PySR ready (min_samples={theorist.min_samples})")
    print(f"  ✓ Navigator: {navigator.mode} mode")
    
    # Reset environment
    print("\n[2/5] Starting exploration...")
    obs, _ = env.reset(seed=42)
    buffer.add_observation(obs)
    
    print(f"  Initial position: ({obs[0]:.2f}, {obs[1]:.2f}), lux: {obs[2]:.2f}")
    
    # Main loop
    print(f"\n[3/5] Running {steps} steps with PySR updates every {update_freq} steps...")
    
    for step in range(steps):
        # Get predicted lux if model exists (for active learning)
        predicted_lux = None
        if theorist.model is not None:
            current_pos = obs[:2]
            dist = np.linalg.norm(current_pos - np.array([0, 0]))
            predicted_lux = theorist.predict(dist)
        
        # Get direction toward unexplored region
        unexplored_dir = buffer.get_unexplored_direction(obs)
        
        # Get action from navigator (active learning)
        action = navigator.get_action(obs, predicted_lux, unexplored_dir)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Calculate Intrinsic Rewards (for monitoring/logging)
        # This simulates what the PPO agent will optimize in Phase 3
        coverage_rew = buffer.get_coverage_reward(obs[0], obs[1])
        intrinsic_rew = navigator.calculate_intrinsic_reward(obs, predicted_lux, coverage_rew)
        total_reward = reward + intrinsic_rew  # Env Reward (Safety) + Intrinsic (Cognitive)
        
        # Store observation
        buffer.add_observation(obs)
        
        # Periodic updates
        if (step + 1) % update_freq == 0:
            coverage = buffer.get_coverage_percentage()
            print(f"\n  Step {step + 1}/{steps}:")
            print(f"    Buffer: {buffer.size()} samples, Coverage: {coverage:.1f}%")
            print(f"    Position: ({obs[0]:.2f}, {obs[1]:.2f}), Lux: {obs[2]:.2f}")
            print(f"    Reward: {total_reward:.2f} (Env: {reward:.1f}, Cov: {coverage_rew:.2f}, Curio: {intrinsic_rew-coverage_rew:.2f})")
            
            # Attempt symbolic regression if enough samples
            if buffer.size() >= theorist.min_samples:
                distances, intensities = buffer.get_distance_intensity_pairs(source_pos=(0, 0))
                
                print(f"    Running PySR on {len(distances)} samples...")
                success = theorist.fit(distances, intensities)
                
                if success:
                    summary = theorist.get_hypothesis_summary()
                    print(f"    Current Hypothesis: {summary['equation']}")
                    print(f"    R² = {summary['r_squared']:.4f}, Complexity = {summary['complexity']}")
                    
                    # Switch to active mode after first hypothesis
                    if navigator.mode == "random" and mode == "active":
                        navigator.set_mode("active")
                        print(f"    ✓ Switching to active learning mode")
            else:
                print(f"    Waiting for {theorist.min_samples} samples before PySR...")
        
        # Small delay if rendering
        if render:
            time.sleep(0.01)
    
    # Final analysis
    print("\n[4/5] Final Analysis...")
    stats = buffer.get_statistics()
    print(f"  ✓ Collected {stats['size']} observations")
    print(f"  Position coverage: x=[{stats['x_mean']:.2f}±{stats['x_std']:.2f}]m")
    print(f"  Position coverage: y=[{stats['y_mean']:.2f}±{stats['y_std']:.2f}]m")
    print(f"  Lux range: [{stats['lux_min']:.2f}, {stats['lux_max']:.2f}]")
    
    print(f"\n  Final Hypothesis: {theorist}")
    
    # Check if hypothesis is good enough
    if theorist.has_good_hypothesis():
        print(f"\n  🎉 Strong hypothesis found! Ready for Phase 3 (LLM validation)")
    else:
        print(f"\n  📊 Need more data or better exploration for strong hypothesis")
    
    # Cleanup
    print("\n[5/5] Cleanup...")
    env.close()
    
    print("\n" + "=" * 60)
    print("Phase 2 Complete! Scientist engine is working.")
    print("=" * 60)
    
    return buffer, theorist


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
        help='Execution phase: 1=Visualization, 2=Scientist Loop, 3=Active Inference (future)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Show MuJoCo viewer visualization'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--update-freq',
        type=int,
        default=100,
        help='Steps between PySR updates (Phase 2)'
    )
    parser.add_argument(
        '--save-csv',
        action='store_true',
        help='Save collected data to CSV'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='active',
        choices=['random', 'active', 'curious'],
        help='Navigation mode for Phase 2: random, active (coverage+surprise), curious (surprise only)'
    )
    
    args = parser.parse_args()
    
    if args.phase == 1:
        phase1_visualization(render=args.render, steps=args.steps)
    elif args.phase == 2:
        phase2_scientist_loop(steps=args.steps, update_freq=args.update_freq, 
                             render=args.render, mode=args.mode)
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
