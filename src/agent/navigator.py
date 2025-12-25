"""
Navigator - Curiosity Controller

This module determines the rover's actions based on exploration strategy.
For Phase 2: Simple random walk
For Phase 3: Surprise-based active learning (curiosity-driven)
"""

# Standard library imports
from typing import Optional, Tuple

# Third-party imports
import numpy as np


class Navigator:
    """
    Controls rover navigation strategy.
    
    Phase 2: Random walk exploration
    Phase 3: Surprise-based active learning (curiosity-driven)
    
    Attributes:
        mode: Current navigation mode ('random' or 'curious')
        surprise_threshold: Threshold for entering "Science Mode"
        science_mode: Whether currently in detailed sampling mode
    """
    
    def __init__(self, mode: str = "random", surprise_threshold: float = 0.15):
        """Initialize navigator.
        
        Args:
            mode: Navigation mode - 'random' for random walk, 'curious' for surprise-based
            surprise_threshold: Threshold for triggering Science Mode (Phase 3)
        """
        self.mode = mode
        self.surprise_threshold = surprise_threshold
        self.science_mode = False
        self.science_samples_remaining = 0
    
    def get_action(self, observation: Optional[np.ndarray] = None, 
                   predicted_lux: Optional[float] = None) -> np.ndarray:
        """Get next action based on navigation strategy.
        
        Args:
            observation: Current observation [x, y, lux]
            predicted_lux: Predicted lux from theorist model (for curiosity mode)
            
        Returns:
            Action array [force_x, force_y] in range [-1, 1]
        """
        if self.mode == "random":
            return self._random_walk()
        elif self.mode == "curious":
            return self._curiosity_driven(observation, predicted_lux)
        else:
            return self._random_walk()
    
    def _random_walk(self) -> np.ndarray:
        """Generate random walk action.
        
        Returns:
            Random force vector in range [-1, 1]
        """
        return np.random.uniform(-1, 1, size=2).astype(np.float32)
    
    def _curiosity_driven(self, observation: Optional[np.ndarray], 
                         predicted_lux: Optional[float]) -> np.ndarray:
        """Surprise-based exploration (Phase 3).
        
        Algorithm from PLAN.md 4.3:
        1. Calculate surprise = |predicted - actual|
        2. If surprise > threshold: Enter Science Mode
        3. In Science Mode: Sample 5 points in 10cm radius
        4. Otherwise: Continue random walk
        
        Args:
            observation: Current [x, y, lux]
            predicted_lux: Model's prediction for current position
            
        Returns:
            Action based on surprise level
        """
        # If no prediction available, fall back to random walk
        if observation is None or predicted_lux is None:
            return self._random_walk()
        
        actual_lux = float(observation[2])
        surprise = abs(predicted_lux - actual_lux)
        
        # Check if we should enter Science Mode
        if surprise > self.surprise_threshold and not self.science_mode:
            print(f"⚠  High surprise detected: {surprise:.3f} > {self.surprise_threshold}")
            print(f"   Predicted: {predicted_lux:.2f}, Actual: {actual_lux:.2f}")
            print(f"   Entering Science Mode: sampling 5 points in 10cm radius")
            self.science_mode = True
            self.science_samples_remaining = 5
        
        # If in Science Mode, sample nearby
        if self.science_mode:
            action = self._sample_nearby()
            self.science_samples_remaining -= 1
            
            if self.science_samples_remaining <= 0:
                print(f"✓  Science Mode complete. Resuming exploration.")
                self.science_mode = False
            
            return action
        
        # Otherwise, continue exploration
        return self._random_walk()
    
    def _sample_nearby(self, radius: float = 0.10) -> np.ndarray:
        """Generate small random movement for detailed sampling.
        
        Args:
            radius: Maximum movement radius in meters
            
        Returns:
            Small random action for nearby sampling
        """
        # Small random movement (scaled down for 10cm radius sampling)
        action = np.random.uniform(-0.2, 0.2, size=2).astype(np.float32)
        return action
    
    def calculate_surprise(self, predicted: float, actual: float) -> float:
        """Calculate prediction error (surprise metric).
        
        Args:
            predicted: Model's predicted lux value
            actual: Observed lux from sensor
            
        Returns:
            Absolute difference between predicted and actual
        """
        return abs(predicted - actual)
    
    def should_enter_science_mode(self, predicted: float, actual: float) -> bool:
        """Determine if anomaly warrants detailed investigation.
        
        Args:
            predicted: Predicted lux value
            actual: Actual lux value
            
        Returns:
            True if surprise exceeds threshold
        """
        surprise = self.calculate_surprise(predicted, actual)
        return surprise > self.surprise_threshold
    
    def reset_science_mode(self):
        """Reset science mode state."""
        self.science_mode = False
        self.science_samples_remaining = 0
    
    def set_mode(self, mode: str):
        """Change navigation mode.
        
        Args:
            mode: 'random' or 'curious'
        """
        self.mode = mode
        self.reset_science_mode()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Navigator(mode={self.mode}, science_mode={self.science_mode})"
