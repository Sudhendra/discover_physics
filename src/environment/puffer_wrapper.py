"""
PufferLib Environment Wrapper

This module contains the gymnasium environment for the Lux Scientia simulation.
It provides the agent's interface to the world through noisy sensors.
"""

# Standard library imports
from typing import Tuple, Optional, Dict, Any

# Third-party imports
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium

# Local imports
from src.environment.physics_engine import LightPhysics


class LuxEnvironment(gymnasium.Env):
    """
    Gymnasium environment for the Lux Scientia robot scientist.
    
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
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render: bool = False, max_ticks: int = 500):
        """Initialize the Lux environment.
        
        Args:
            render: If True, show PyBullet GUI visualization
            max_ticks: Maximum steps per episode
        """
        super().__init__()
        
        # 1. Physics Engine Initialization
        self.render_mode = "human" if render else None
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1/240.0, physicsClientId=self.client)  # 240 Hz
        
        # 2. Asset Loading
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        # Create a simple box as rover (since rover.urdf may not exist)
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], 
                                          rgbaColor=[0.8, 0.2, 0.2, 1.0])
        self.rover_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[5, 5, 0.1],
            physicsClientId=self.client
        )
        
        # 3. Internal Truth Instantiation
        self.truth = LightPhysics(source_pos=(0, 0), source_intensity=1000.0)
        
        # 4. Simulation Constraints
        self.tick = 0
        self.max_ticks = max_ticks
        
        # 5. Define observation and action spaces
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 5000], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
    
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
        
        p.resetBasePositionAndOrientation(
            self.rover_id, 
            [start_x, start_y, 0.1], 
            [0, 0, 0, 1],
            physicsClientId=self.client
        )
        
        # Reset velocity to zero
        p.resetBaseVelocity(
            self.rover_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.client
        )
        
        self.tick = 0
        
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
        force_x, force_y = float(action[0]) * 10.0, float(action[1]) * 10.0
        p.applyExternalForce(
            self.rover_id, 
            -1, 
            [force_x, force_y, 0], 
            [0, 0, 0], 
            p.WORLD_FRAME,
            physicsClientId=self.client
        )
        
        # Step physics simulation
        p.stepSimulation(physicsClientId=self.client)
        
        obs = self._get_obs()
        
        # Reward Engineering:
        # Initially: simple exploration reward (distance covered or new area)
        # Later Phase: Information Gain (Entropy reduction)
        reward = 0.0
        
        # Episode termination
        done = self.tick >= self.max_ticks
        truncated = False
        
        # Autoreset for PufferLib compatibility
        if done or truncated:
            obs, _ = self.reset()
        
        return obs, reward, done, truncated, {}
    
    def _get_obs(self) -> np.ndarray:
        """Get noisy observation from sensors.
        
        Returns:
            Array [x, y, lux] with sensor noise applied
        """
        # 1. Ground Truth Extraction
        pos, _ = p.getBasePositionAndOrientation(self.rover_id, physicsClientId=self.client)
        raw_x, raw_y = float(pos[0]), float(pos[1])
        
        # 2. Noise Injection (Simulating Sensor Imperfection)
        # Position noise: ±1cm standard deviation
        obs_x = raw_x + np.random.normal(0, 0.01)
        obs_y = raw_y + np.random.normal(0, 0.01)
        
        # Light sensor noise: ±5% standard deviation
        true_lux = self.truth.get_true_lux(np.array([raw_x, raw_y]))
        obs_lux = true_lux * (1 + np.random.normal(0, 0.05))
        
        return np.array([obs_x, obs_y, obs_lux], dtype=np.float32)
    
    def close(self):
        """Clean up PyBullet resources."""
        if hasattr(self, 'client'):
            p.disconnect(physicsClientId=self.client)
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
