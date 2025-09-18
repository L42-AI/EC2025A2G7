import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load npz
data = np.load("A2/results/EA1_run01.npz")
gen = data["gen"]
avg = data["avg"]
std = data["std"]
minv = data["min"]
maxv = data["max"]

# Build tidy dataframe
df = pd.DataFrame({
    "Generation": gen,
    "Average": avg,
    "Std": std,
    "Min": minv,
    "Max": maxv
})

# Moving average (window=3 generations as example)
window = 3
df["MovingAvg"] = df["Average"].rolling(window=window, center=True).mean()
df["MovingStd"] = df["Average"].rolling(window=window, center=True).std()

# Plot
plt.figure(figsize=(10,6))

# Scatter of raw avg values per generation
sns.scatterplot(data=df, x="Generation", y="Average", s=30, color="blue", alpha=0.5, label="Fitness")

# Moving average line
sns.lineplot(data=df, x="Generation", y="MovingAvg", color="purple", linewidth=2, label="Moving average")

# Shaded region = moving average ± std
plt.fill_between(df["Generation"],
                 df["MovingAvg"] - df["MovingStd"],
                 df["MovingAvg"] + df["MovingStd"],
                 alpha=0.2, color="red", label="moving ± std")

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Evolution Progress with Moving Average")
plt.legend()
plt.tight_layout()
plt.show()
