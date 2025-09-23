import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(series, window=3):
    """Return moving average and std for a 1D array or pandas Series."""
    s = pd.Series(series)
    ma = s.rolling(window=window, center=True, min_periods=1).mean()
    ms = s.rolling(window=window, center=True, min_periods=1).std()
    return ma, ms

def plot_multiple_results(npy_paths):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "red", "green"]
    for i, path in enumerate(npy_paths):
        arr = np.load(path).ravel()
        generations = np.arange(len(arr))

        mov_avg, mov_std = moving_average(arr)

        # Scatter (raw fitness points)
        sns.scatterplot(x=generations, y=arr,
                        s=20, alpha=0.4, color=colors[i],
                        label=f"Fitness run {i+1}", ax=ax)

        # Moving average line
        sns.lineplot(x=generations, y=mov_avg,
                     linewidth=2, color=colors[i],
                     label=f"Moving avg run {i+1}", ax=ax)

        # Shaded std band
        ax.fill_between(generations,
                        mov_avg - mov_std,
                        mov_avg + mov_std,
                        alpha=0.15, color=colors[i])

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness over Generations (Baseline NPY files)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    paths = [
        "A2/baseline_results/13_baseline_fitnesses.npy",
        "A2/baseline_results/24_baseline_fitnesses.npy",
        "A2/baseline_results/42_baseline_fitnesses.npy",
    ]
    plot_multiple_results(paths)