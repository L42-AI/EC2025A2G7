from typing import Optional
import numpy as np

__all__ = [
    "xy_displacement",
    "x_speed",
    "y_speed",
    "distance_to_target",
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
    return ((xyz1[0] - xyz2[0]) ** 2 + (xyz1[1] - xyz2[1]) ** 2 + (xyz1[2] - xyz2[2]) ** 2) ** 0.5

def xy_displacement(xy1, xy2) -> float:
    """
    Calculate the displacement between two points in 2D space.

    Parameters:
    xy1 (tuple): Coordinates of the first point (x1, y1).
    xy2 (tuple): Coordinates of the second point (x2, y2).

    Returns:
    float: The Euclidean distance between the two points.
    """
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

def x_speed(xy1, xy2, dt) -> float:
    """
    Calculate the speed in the x direction between two points.

    Parameters:
    xy1 (tuple): Coordinates of the first point (x1, y1).
    xy2 (tuple): Coordinates of the second point (x2, y2).
    dt (float): Time difference between the two points.

    Returns:
    float: The speed in the x direction.
    """
    return abs(xy2[0] - xy1[0]) / dt if dt > 0 else 0.0

def y_speed(xy1, xy2, dt) -> float:
    """
    Calculate the speed in the y direction between two points.

    Parameters:
    xy1 (tuple): Coordinates of the first point (x1, y1).
    xy2 (tuple): Coordinates of the second point (x2, y2).
    dt (float): Time difference between the two points.

    Returns:
    float: The speed in the y direction.
    """
    return abs(xy2[1] - xy1[1]) / dt if dt > 0 else 0.0

def distance_to_target(initial_position, target_position):
    """
    Calculate the Euclidean distance between the current position and the target position.

    Args:
        initial_position (tuple): The current position as (x, y).
        target_position (tuple): The target position as (x, y).

    Returns:
        float: The distance between the two positions.
    """
    return (
        (initial_position[0] - target_position[0]) ** 2
        + (initial_position[1] - target_position[1]) ** 2
    ) ** 0.5

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

def get_furthest_xy_distance(history: list, target: Optional[np.ndarray] = None) -> float:
    """Calculate and print the furthest point reached in the XY plane."""
    if target is None:
        target = history[0][:2]
    
    max_distance = 0.0
    for step in history:
        displacement = xy_displacement(target, step[:2])

        if displacement > max_distance:
            max_distance = displacement

    return max_distance

def get_furthest_xyz_distance(history: list, target: Optional[np.ndarray] = None) -> float:
    """Calculate and print the furthest point reached in the XYZ plane."""
    if target is None:
        target = history[0][:3]

    return xyz_displacement(target, history[-1][:3])

def get_target_fitness(history: list, target: Optional[np.ndarray] = None) -> float:
    """Sum of all step-to-step displacements (path length)."""
    if target is None:
        target = history[0][:3]

    # --- 1. Target distance ---
    target_distance = xyz_displacement(target, history[-1][:3])
    
    # --- 2. Total movement ---
    total_disp = sum(
        xyz_displacement(history[i][:3], history[i+1][:3]) 
        for i in range(len(history)-1)
    )

    return target_distance - 0.2 * total_disp