"""
Physics Engine - The Truth Engine

This module contains the hidden physical laws that govern the simulated world.
The agent must discover these laws through experimentation without direct access.
"""

# Standard library imports
from typing import Tuple

# Third-party imports
import numpy as np


class LightPhysics:
    """
    The 'Hidden Truth' of the universe.
    
    Implements the Inverse Square Law of Light: I = S / (4 * pi * r^2)
    The agent must derive this logic via experimentation with noisy sensors.
    
    Attributes:
        source_pos: Position of the light source (x, y) in meters
        source_intensity: Light source intensity in lux at 1 meter
    """
    
    def __init__(self, source_pos: Tuple[float, float] = (0, 0), source_intensity: float = 1000.0):
        """Initialize the light physics engine.
        
        Args:
            source_pos: (x, y) position of light source in meters
            source_intensity: Base intensity of light source in lux
        """
        self.source_pos = np.array(source_pos, dtype=np.float32)
        self.source_intensity = source_intensity
    
    def get_true_lux(self, rover_pos: np.ndarray) -> float:
        """Calculate true light intensity at rover position.
        
        Implements the Inverse Square Law: I = S / (4 * pi * r^2)
        
        Args:
            rover_pos: (x, y) position of the rover in meters
            
        Returns:
            Light intensity in lux at the rover's position
            
        Note:
            Includes singularity protection to prevent division by zero
            when rover is very close to the light source.
        """
        # Calculate squared distance to light source
        dist_sq = np.sum((rover_pos - self.source_pos)**2)
        
        # Singularity protection: clamp minimum distance to 10cm
        dist_sq = max(float(dist_sq), 0.01)
        
        # Inverse square law
        return float(self.source_intensity / (4 * np.pi * dist_sq))
