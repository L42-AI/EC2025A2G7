import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_file(name:str):
    data = np.load(name)
    print(data.files)

    gen = data["gen"]
    avg = data["avg"]
    std = data["std"]
    minv = data["min"]
    maxv = data["max"]

    print("Generations:", gen[:5], "...")
    print("Avg fitness:", avg[:5], "...")
    return gen, avg, std, minv, maxv

gen, avg, std, minv, maxv = load_file("A2/results/EA1_run01.npz")


def plot_fitness(gen, avg, std, minv, maxv):
    plt.figure(figsize=(8, 5))
    plt.plot(gen, avg, label="average fitness")
    plt.plot(gen, maxv, label="Max fitness")
    plt.plot(gen, minv, label="Min fitness")

    # Optional: add error bars with std
    plt.fill_between(gen, avg - std, avg + std, color="gray", alpha=0.2, label="Std dev")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution Progress (from npz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_fitness(gen, avg, std, minv, maxv)






