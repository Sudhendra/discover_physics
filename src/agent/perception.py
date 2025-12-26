"""
Perception Buffer - Data Collection and Preprocessing

This module manages the rolling buffer of observations collected during exploration.
It stores (x, y, lux) tuples and can preprocess them for symbolic regression.
"""

# Standard library imports
from typing import List, Tuple, Optional
from collections import deque

# Third-party imports
import numpy as np


class PerceptionBuffer:
    """
    Rolling buffer for storing exploration history.
    
    Stores observations as (x, y, lux) tuples and provides preprocessing
    for symbolic regression (e.g., calculating distance from hypothesized origin).
    
    Attributes:
        max_size: Maximum number of observations to store
        buffer: Deque of (x, y, lux) tuples
        source_hypothesis: Hypothesized light source position (x, y)
    """
    
    def __init__(self, max_size: int = 1000, grid_size: int = 10):
        """Initialize perception buffer.
        
        Args:
            max_size: Maximum buffer size (oldest samples discarded when full)
            grid_size: Grid resolution for coverage tracking (10 = 10x10 cells)
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.source_hypothesis: Optional[np.ndarray] = None
        
        # Coverage tracking for active learning
        self.grid_size = grid_size
        self.visit_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.arena_bounds = (0.0, 10.0)  # Arena from [0, 10] x [0, 10]
    
    def add(self, x: float, y: float, lux: float):
        """Add observation to buffer.
        
        Args:
            x: X position in meters
            y: Y position in meters
            lux: Light intensity reading
        """
        self.buffer.append((x, y, lux))
        
        # Update coverage tracking
        self._update_coverage(x, y)
    
    def add_observation(self, obs: np.ndarray):
        """Add observation from environment step.
        
        Args:
            obs: Observation array [x, y, lux]
        """
        self.add(float(obs[0]), float(obs[1]), float(obs[2]))
    
    def get_data(self) -> np.ndarray:
        """Get all buffered data as numpy array.
        
        Returns:
            Array of shape (n, 3) with columns [x, y, lux]
        """
        if len(self.buffer) == 0:
            return np.array([]).reshape(0, 3)
        return np.array(list(self.buffer), dtype=np.float32)
    
    def get_distance_intensity_pairs(self, source_pos: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get (distance, intensity) pairs for regression.
        
        Args:
            source_pos: Light source position (x, y). If None, uses (0, 0)
            
        Returns:
            Tuple of (distances, intensities) arrays
            
        Note:
            Preprocessing step that converts (x, y, lux) to (distance, lux)
            for easier symbolic regression.
        """
        if len(self.buffer) == 0:
            return np.array([]), np.array([])
        
        data = self.get_data()
        positions = data[:, :2]  # (x, y) columns
        intensities = data[:, 2]  # lux column
        
        # Use provided source position or default to (0, 0)
        if source_pos is None:
            if self.source_hypothesis is not None:
                source = self.source_hypothesis
            else:
                source = np.array([0, 0], dtype=np.float32)
        else:
            source = np.array(source_pos, dtype=np.float32)
        
        # Calculate Euclidean distances
        distances = np.linalg.norm(positions - source, axis=1)
        
        return distances, intensities
    
    def set_source_hypothesis(self, x: float, y: float):
        """Set hypothesized light source position.
        
        Args:
            x: Hypothesized x position of light source
            y: Hypothesized y position of light source
            
        Note:
            This is set by the theorist when it discovers the source location.
            Initially unknown, can be discovered through analysis.
        """
        self.source_hypothesis = np.array([x, y], dtype=np.float32)
    
    def size(self) -> int:
        """Get current buffer size.
        
        Returns:
            Number of observations in buffer
        """
        return len(self.buffer)
    
    def clear(self):
        """Clear all buffered data."""
        self.buffer.clear()
        self.visit_counts.fill(0)
    
    def _update_coverage(self, x: float, y: float):
        """Update coverage grid with new position.
        
        Args:
            x: X position
            y: Y position
        """
        # Convert position to grid cell
        cell_x = int((x - self.arena_bounds[0]) / (self.arena_bounds[1] - self.arena_bounds[0]) * self.grid_size)
        cell_y = int((y - self.arena_bounds[0]) / (self.arena_bounds[1] - self.arena_bounds[0]) * self.grid_size)
        
        # Clamp to valid range
        cell_x = np.clip(cell_x, 0, self.grid_size - 1)
        cell_y = np.clip(cell_y, 0, self.grid_size - 1)
        
        self.visit_counts[cell_x, cell_y] += 1
    
    def get_unexplored_direction(self, current_pos: np.ndarray) -> Optional[np.ndarray]:
        """Get direction toward least-visited region.
        
        Args:
            current_pos: Current position [x, y]
            
        Returns:
            Unit vector toward unexplored region, or None if all explored
        """
        if np.all(self.visit_counts > 0):
            # All cells visited - return None
            return None
        
        # Find least-visited cell
        min_visits = np.min(self.visit_counts[self.visit_counts >= 0])
        unvisited_cells = np.argwhere(self.visit_counts == min_visits)
        
        if len(unvisited_cells) == 0:
            return None
        
        # Pick random unvisited cell
        target_cell = unvisited_cells[np.random.randint(len(unvisited_cells))]
        
        # Convert cell to world position (cell center)
        cell_width = (self.arena_bounds[1] - self.arena_bounds[0]) / self.grid_size
        target_x = self.arena_bounds[0] + (target_cell[0] + 0.5) * cell_width
        target_y = self.arena_bounds[0] + (target_cell[1] + 0.5) * cell_width
        
        # Direction vector
        direction = np.array([target_x, target_y]) - current_pos[:2]
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            return direction / norm
        return None
    
    def get_coverage_percentage(self) -> float:
        """Get percentage of arena explored.
        
        Returns:
            Percentage of grid cells visited (0-100)
        """
        visited = np.sum(self.visit_counts > 0)
        total = self.grid_size * self.grid_size
        return float(100.0 * visited / total)
    
    def get_statistics(self) -> dict:
        """Get summary statistics of buffered data.
        
        Returns:
            Dictionary with mean, std, min, max for each dimension
        """
        if len(self.buffer) == 0:
            return {"size": 0}
        
        data = self.get_data()
        
        return {
            "size": len(self.buffer),
            "x_mean": float(np.mean(data[:, 0])),
            "x_std": float(np.std(data[:, 0])),
            "y_mean": float(np.mean(data[:, 1])),
            "y_std": float(np.std(data[:, 1])),
            "lux_mean": float(np.mean(data[:, 2])),
            "lux_std": float(np.std(data[:, 2])),
            "lux_min": float(np.min(data[:, 2])),
            "lux_max": float(np.max(data[:, 2])),
        }
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return f"PerceptionBuffer(size={stats.get('size', 0)}, lux_range=[{stats.get('lux_min', 0):.2f}, {stats.get('lux_max', 0):.2f}])"
