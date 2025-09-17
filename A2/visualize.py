import numpy as np
import matplotlib.pyplot as plt

def visualise_furthest_point(furthest_points: list):
    """Visualise the frequency of furthest points reached in multiple simulations as a histogram."""

    plt.figure(figsize=(10, 6))
    plt.hist(furthest_points, bins='auto', edgecolor='black')
    plt.title('Frequency of Furthest Points Reached in Simulations')
    plt.xlabel('Furthest Point (XY Plane)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('furthest_points_histogram.png')
    plt.show()

def show_qpos_history(history:list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.show()
