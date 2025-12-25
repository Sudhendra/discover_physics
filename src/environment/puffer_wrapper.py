"""
MuJoCo Environment Wrapper

This module contains the gymnasium environment for the Lux Scientia simulation.
It provides the agent's interface to the world through noisy sensors.
"""

# Standard library imports
from typing import Tuple, Optional, Dict, Any

# Third-party imports
import numpy as np
import mujoco
import gymnasium
from gymnasium import spaces

# Local imports
from src.environment.physics_engine import LightPhysics


class LuxEnvironment(gymnasium.Env):
    """
    Gymnasium environment for the Lux Scientia robot scientist using MuJoCo.
    
    The rover explores a 10m x 10m arena trying to discover the inverse square
    law of light through noisy sensor measurements.
    
    Observation Space:
        Box(3,): [x_position, y_position, lux_reading]
        - x, y: Position in meters (±1cm noise)
        - lux: Light intensity in lux (±5% noise)
    
    Action Space:
        Box(2,): [force_x, force_y]
        - Normalized forces in range [-1, 1]
        - Scaled to Newtons internally
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None, max_ticks: int = 500):
        """Initialize the Lux environment.
        
        Args:
            render_mode: "human" for visualization, None for headless
            max_ticks: Maximum steps per episode
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_ticks = max_ticks
        
        # 1. Create MuJoCo model programmatically
        self.model = self._create_mujoco_model()
        self.data = mujoco.MjData(self.model)
        
        # 2. Setup viewer if rendering
        self.viewer = None
        if render_mode == "human":
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        # 3. Internal Truth Instantiation
        self.truth = LightPhysics(source_pos=(0, 0), source_intensity=1000.0)
        
        # 4. Simulation state
        self.tick = 0
        
        # 5. Define observation and action spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 5000], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
    
    def _create_mujoco_model(self) -> mujoco.MjModel:
        """Create a simple MuJoCo model for 2D rover simulation.
        
        Returns:
            MuJoCo model with rover and plane
        """
        xml_string = """
        <mujoco model="lux_rover">
            <option gravity="0 0 -9.81" timestep="0.01" integrator="RK4"/>
            
            <visual>
                <global offwidth="1280" offheight="720"/>
                <quality shadowsize="4096"/>
            </visual>
            
            <worldbody>
                <light pos="5 5 10" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
                
                <!-- Floor centered at (5, 5, 0) to cover [0,10] x [0,10] arena -->
                <geom name="floor" type="plane" size="5 5 0.1" pos="5 5 0" 
                      rgba="0.9 0.9 0.9 1" friction="0.5 0.005 0.0001"/>
                
                <!-- Visual marker at light source (0, 0) for debugging -->
                <body name="light_source_marker" pos="0 0 0.05">
                    <geom name="source_marker" type="sphere" size="0.15" 
                          rgba="1 1 0 0.8" contype="0" conaffinity="0"/>
                </body>
                
                <!-- Rover (starts at random position in [1,9] x [1,9]) -->
                <body name="rover" pos="5 5 0.1">
                    <freejoint name="rover_root"/>
                    <geom name="rover_body" type="box" size="0.1 0.1 0.05" 
                          mass="1.0" rgba="0.8 0.2 0.2 1"/>
                    <site name="rover_site" pos="0 0 0" size="0.02"/>
                </body>
            </worldbody>
            
            <actuator>
                <general name="force_x" joint="rover_root" gear="1 0 0 0 0 0" dyntype="none" gaintype="fixed" biastype="none" ctrlrange="-10 10"/>
                <general name="force_y" joint="rover_root" gear="0 1 0 0 0 0" dyntype="none" gaintype="fixed" biastype="none" ctrlrange="-10 10"/>
            </actuator>
        </mujoco>
        """
        return mujoco.MjModel.from_xml_string(xml_string)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Stochastic reset to prevent overfitting to a specific start path
        if seed is not None:
            np.random.seed(seed)
        
        start_x = np.random.uniform(1, 9)
        start_y = np.random.uniform(1, 9)
        
        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial position (qpos indices depend on freejoint: [x, y, z, qw, qx, qy, qz])
        self.data.qpos[0] = start_x  # x position
        self.data.qpos[1] = start_y  # y position
        self.data.qpos[2] = 0.1      # z position (slightly above ground)
        self.data.qpos[3] = 1.0      # quaternion w
        self.data.qpos[4:7] = 0.0    # quaternion x, y, z
        
        # Reset velocity to zero
        self.data.qvel[:] = 0.0
        
        self.tick = 0
        
        # Forward kinematics to update state
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Args:
            action: [force_x, force_y] in range [-1, 1]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.tick += 1
        
        # Clamp actions to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Action -> Force Application
        # Scaling factor 10.0 converts abstract action to Newtons
        self.data.ctrl[0] = float(action[0]) * 10.0  # force_x
        self.data.ctrl[1] = float(action[1]) * 10.0  # force_y
        
        # Step MuJoCo physics
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        # Reward Engineering:
        # Initially: simple exploration reward (distance covered or new area)
        # Later Phase: Information Gain (Entropy reduction)
        reward = 0.0
        
        # Episode termination
        terminated = self.tick >= self.max_ticks
        truncated = False
        
        # Render if needed
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.render()
        
        # Autoreset for PufferLib compatibility
        if terminated or truncated:
            obs, _ = self.reset()
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self) -> np.ndarray:
        """Get noisy observation from sensors.
        
        Returns:
            Array [x, y, lux] with sensor noise applied
        """
        # 1. Ground Truth Extraction from MuJoCo
        raw_x = float(self.data.qpos[0])
        raw_y = float(self.data.qpos[1])
        
        # 2. Noise Injection (Simulating Sensor Imperfection)
        # Position noise: ±1cm standard deviation
        obs_x = raw_x + np.random.normal(0, 0.01)
        obs_y = raw_y + np.random.normal(0, 0.01)
        
        # Light sensor noise: ±5% standard deviation
        true_lux = self.truth.get_true_lux(np.array([raw_x, raw_y]))
        obs_lux = true_lux * (1 + np.random.normal(0, 0.05))
        
        return np.array([obs_x, obs_y, obs_lux], dtype=np.float32)
    
    def close(self):
        """Clean up MuJoCo resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
