from fitness_functions import xy_displacement

def get_furthest_point(history: list):
    """Calculate and print the furthest point reached in the XY plane."""
    max_distance = 0.0
    for step in history:
        displacement = xy_displacement(history[0][:2], step[:2])

        if displacement > max_distance:
            max_distance = displacement

    return max_distance