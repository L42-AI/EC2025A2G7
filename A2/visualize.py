import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, glob


def visualise_furthest_point(furthest_points: list):
    """Visualise the frequency of furthest points reached in multiple simulations as a histogram."""

    plt.figure(figsize=(10, 6))
    plt.hist(furthest_points, bins="auto", edgecolor="black")
    plt.title("Frequency of Furthest Points Reached in Simulations")
    plt.xlabel("Furthest Point (XY Plane)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("furthest_points_histogram.png")
    plt.show()


def show_qpos_history(history: list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis("equal")
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    plt.show()


# Load npz
def load_npz_file(path: str) -> dict:
    data = np.load(path)
    out = {k: data[k] for k in data.files}
    return out


# Build pandas dataframe
def build_pandas_dataframe(dict):
    df = pd.DataFrame(
        {
            "Generation": dict["gen"],
            "Average": dict["avg"],
            "Std": dict["std"],
            "Min": dict["min"],
            "Max": dict["max"],
            "Mutpb": dict["mutpb"],
            "Cxpb": dict["cxpb"],
        }
    )
    return df


# Moving average window=3 generations
def moving_average(df):
    window = 3
    df["MovingAvg"] = (
        df["Average"].rolling(min_periods=1, window=window, center=True).mean()
    )
    df["MovingStd"] = (
        df["Average"].rolling(min_periods=1, window=window, center=True).std()
    )
    return df


def plot_result_mutcx_pb(path: str, name: str):
    d = load_npz_file(path)
    df = moving_average(build_pandas_dataframe(d))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    fig.canvas.manager.set_window_title(name)

    sns.scatterplot(
        data=df,
        x="Generation",
        y="Average",
        ax=ax1,
        s=30,
        alpha=0.5,
        color="blue",
        label="Fitness",
    )
    sns.lineplot(
        data=df,
        x="Generation",
        y="MovingAvg",
        ax=ax1,
        linewidth=2,
        color="purple",
        label="Moving average",
    )
    ax1.fill_between(
        df["Generation"],
        df["MovingAvg"] - df["MovingStd"],
        df["MovingAvg"] + df["MovingStd"],
        alpha=0.2,
        color="orange",
        label="Moving ± std",
    )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Fitness over Generations")
    ax1.legend()

    sns.lineplot(
        data=df,
        x="Generation",
        y="Mutpb",
        ax=ax2,
        linewidth=2,
        color="green",
        label="Mutation prob",
    )
    sns.lineplot(
        data=df,
        x="Generation",
        y="Cxpb",
        ax=ax2,
        linewidth=2,
        color="black",
        label="Crossover prob",
    )
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Probability")
    ax2.set_title("Evolutionary Parameters")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def df_from_run(path) -> pd.DataFrame:
    filetype = os.path.splitext(path)[1].lower()
    if filetype == ".npz":
        d = load_npz_file(path)
        df = build_pandas_dataframe(d)
        return df[["Generation", "Average"]].copy()
    elif filetype == ".npy":
        arr = np.load(path)
        return pd.DataFrame(
            {"Generation": np.arange(len(arr), dtype=int), "Average": arr}
        )
    else:
        raise ValueError(f"Unsupported file type for {path}. Use .npz or .npy")


def plot_agg_results(
    path_1: str,
    path_2: str,
    path_3: str,
    path_4: str,
    ax,
    title: str = "Fitness over generations",
):
    dfs = [df_from_run(p) for p in (path_1, path_2, path_3, path_4)]
    all_df = pd.concat(dfs, ignore_index=True)

    agg = all_df.groupby("Generation", as_index=False)["Average"].agg(
        Mean="mean", Std="std"
    )

    tmp = agg.rename(columns={"Mean": "Average"})
    mov = moving_average(tmp)
    df_plot = mov.merge(agg[["Generation", "Std"]], on="Generation", how="left")

    sns.scatterplot(
        data=df_plot,
        x="Generation",
        y="Average",
        ax=ax,
        s=30,
        alpha=0.5,
        color="blue",
        label="Fitness (mean)",
    )
    sns.lineplot(
        data=df_plot,
        x="Generation",
        y="MovingAvg",
        ax=ax,
        linewidth=2,
        color="purple",
        label="Moving average",
    )
    ax.fill_between(
        df_plot["Generation"],
        df_plot["MovingAvg"] - df_plot["MovingStd"],
        df_plot["MovingAvg"] + df_plot["MovingStd"],
        alpha=0.2,
        color="orange",
        label="Moving ± std",
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.legend(loc="best")
    return df_plot


def plot_3_experiments(
    baseline_exp,
    regular_ea,
    enhanced_ea,
    titles=("Baseline", "Standard EA", "Elevated EA"),
    window_title="Aggregated fitness",
):
    """
    triplet_left/right: (path1, path2, path3) for each condition
    baseline_path: (path1, path2, path3) for each baseline
    """

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass

    plot_agg_results(*baseline_exp, ax=axes[0], title=titles[0])
    plot_agg_results(*regular_ea, ax=axes[1], title=titles[1])
    plot_agg_results(*enhanced_ea, ax=axes[2], title=titles[2])

    plt.tight_layout()
    plt.show()
    return fig, axes


if __name__ == "__main__":
    plot_3_experiments(
        baseline_exp=(
            Path(__file__).parent
            / "baseline_results/123_baseline_fitnesses_new_run.npy",
            Path(__file__).parent
            / "baseline_results/123_baseline_fitnesses_new_run.npy",
            Path(__file__).parent
            / "baseline_results/123_baseline_fitnesses_new_run.npy",
            Path(__file__).parent
            / "baseline_results/123_baseline_fitnesses_new_run.npy",
        ),
        regular_ea=(
            Path(__file__).parent / "ea_results/123_experiment_CL_False_new_run.npz",
            Path(__file__).parent / "ea_results/13_experiment_CL_False_new_run.npz",
            Path(__file__).parent / "ea_results/42_experiment_CL_False_new_run.npz",
            Path(__file__).parent / "ea_results/24_experiment_CL_False_new_run.npz",
        ),
        enhanced_ea=(
            Path(__file__).parent / "ea_results/123_experiment_CL_True_new_run.npz",
            Path(__file__).parent / "ea_results/13_experiment_CL_True_new_run.npz",
            Path(__file__).parent / "ea_results/42_experiment_CL_True_new_run.npz",
            Path(__file__).parent / "ea_results/24_experiment_CL_True_new_run.npz",
        ),
        titles=("Baseline", "Standard EA", "Enhanced EA"),
        window_title="Aggregated fitness",
    )
    # path = Path(__file__).parent / "ea_results/24_experiment_CL_False.npz"
    # plot_result_mutcx_pb(path, "24_experiment_CL_False")
    # path = Path(__file__).parent / "ea_results/24_experiment_CL_True.npz"
    # plot_result_mutcx_pb(path, "24_experiment_CL_True")
