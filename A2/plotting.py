import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
