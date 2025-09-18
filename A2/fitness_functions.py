from typing import Optional
import numpy as np

__all__ = [
    "xy_displacement",
    "x_speed",
    "y_speed",
    "turning_in_place",
]

def xyz_displacement(xyz1, xyz2) -> float:
    """
    Calculate the displacement between two points in 3D space.

    Parameters:
    xyz1 (tuple): Coordinates of the first point (x1, y1, z1).
    xyz2 (tuple): Coordinates of the second point (x2, y2, z2).

    Returns:
    float: The Euclidean distance between the two points.
    """
    return (
        (xyz1[0] - xyz2[0]) ** 2
        + (xyz1[1] - xyz2[1]) ** 2
        + (xyz1[2] - xyz2[2]) ** 2
        ) ** 0.5

def x_speed(xy1, xy2, dt) -> float:
    return abs(xy2[0] - xy1[0]) / dt if dt > 0 else 0.0

def y_speed(xy1, xy2, dt) -> float:
    return abs(xy2[1] - xy1[1]) / dt if dt > 0 else 0.0

def z_speed(z1, z2, dt) -> float:
    return abs(z2 - z1) / dt if dt > 0 else 0.0

def turning_in_place(xy_history) -> float:
    """
    Determines the total angle turned by a robot based on its path history.

    Parameters:
    xy_history (list of tuples): The history of x, y coordinates from a simulation i.e. robot path.

    Returns:
    float: The total angle turned by the robot.
    """

    xy = np.array(xy_history)
    if len(xy) < 2:
        return 0.0

    # Headings from XY positions
    deltas = np.diff(xy, axis=0)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    headings_unwrapped = np.unwrap(headings)

    # Total amount turned (absolute rotation)
    total_turning_angle = np.sum(np.abs(np.diff(headings_unwrapped)))

    # Drift from start position
    displacement = np.linalg.norm(xy[-1] - xy[0])

    # Penalize if robot drifts away
    fitness = total_turning_angle / (1.0 + displacement)

    return fitness

def get_last_distance_from_start(history: list) -> float:
    """Calculate and print the furthest point reached in the XY plane."""
    return xyz_displacement(history[0][:3], history[-1][:3])

def get_best_distance_from_start(history: list) -> float:
    """Calculate and print the furthest point reached in the XY plane."""
    start = history[0][:3]

    max_distance = 0.0
    for state in history:
        distance = xyz_displacement(start, state[:3])

        if distance > max_distance:
            max_distance = distance
        
    return max_distance

def get_end_closeness_to_xyz(history: list, target: Optional[np.ndarray]) -> float:
    """Calculate and print the furthest point reached in the XYZ plane."""
    return xyz_displacement(target, history[-1][:3])

def get_best_closeness_to_xyz(history: list, target: Optional[np.ndarray]) -> float:
    """Calculate and print the furthest point reached in the XYZ plane."""
    if target is None:
        target = history[0][:3]

    min_distance = float('inf')
    for state in history:
        distance = xyz_displacement(target, state[:3])

        if distance < min_distance:
            min_distance = distance
        
    return min_distance