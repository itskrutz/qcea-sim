import matplotlib.pyplot as plt

def plot_metrics(metrics):
    for key, values in metrics.data.items():
        plt.figure()
        plt.plot(values)
        plt.title(f"{key.title()} Over Time")
        plt.xlabel("Time Step")
        plt.ylabel(key.title())
        plt.grid(True)
        plt.tight_layout()
        plt.show()
