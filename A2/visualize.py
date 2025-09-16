import matplotlib.pyplot as plt

def visualise_furthest_point(furthest_points: list):
    """Visualise the frequency of furthest points reached in multiple simulations as a histogram."""

    plt.figure(figsize=(10, 6))
    plt.hist(furthest_points, bins='auto', edgecolor='black')
    plt.title('Frequency of Furthest Points Reached in Simulations')
    plt.xlabel('Furthest Point (XY Plane)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
