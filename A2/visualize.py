import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

# Load npz
def load_npz_file(path:str)-> dict:
    data = np.load(path)
    out = {k: data[k] for k in data.files}
    print("keys:", out.keys())         # Do you see 'mutpb' and 'cxpb'?
    print(out.get("mutpb"), out.get("cxpb"))    
    return out


# Build tidy dataframe
def build_pandas_dataframe( dict ):
    df = pd.DataFrame({
        "Generation": dict["gen"],
        "Average": dict["avg"],
        "Std": dict["std"],
        "Min": dict["min"],
        "Max": dict["max"],
        "Mutpb": dict.get("mutpb", np.full_like(dict["gen"], np.nan, dtype=float)),
        "Cxpb": dict.get("cxpb", np.full_like(dict["gen"], np.nan, dtype=float))
    })
    return df

# Moving average window=3 generations 
def moving_average(df):
    window = 3
    df["MovingAvg"] = df["Average"].rolling(min_periods=1, window=window, center=True).mean()
    df["MovingStd"] = df["Average"].rolling(min_periods=1, window=window, center=True).std()
    return df

def plot_result(path: str):
    d  = load_npz_file(path)
    df = moving_average(build_pandas_dataframe(d))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # --- Fitness plot (left) ---
    sns.scatterplot(data=df, x="Generation", y="Average",
                    ax=ax1, s=30, alpha=0.5, color="blue", label="Fitness")
    sns.lineplot(data=df, x="Generation", y="MovingAvg",
                 ax=ax1, linewidth=2, color="purple", label="Moving average")
    ax1.fill_between(df["Generation"],
                     df["MovingAvg"] - df["MovingStd"],
                     df["MovingAvg"] + df["MovingStd"],
                     alpha=0.2, color="orange", label="Moving ± std")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Fitness over Generations")
    ax1.legend()

    # --- Probabilities plot (right) ---
    sns.lineplot(data=df, x="Generation", y="Mutpb",
                 ax=ax2, linewidth=2, color="green", label="Mutation prob")
    sns.lineplot(data=df, x="Generation", y="Cxpb",
                 ax=ax2, linewidth=2, color="black", label="Crossover prob")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Probability")
    ax2.set_title("Evolutionary Parameters")
    ax2.legend()

    plt.tight_layout()
    plt.show()

plot_result("A2/results/EA1_run01_20250920-220956.npz")
