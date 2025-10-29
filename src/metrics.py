import numpy as np

class Metrics:
    def __init__(self):
        self.data = {
            "delay": [],
            "throughput": [],
            "energy": [],
            "packet_delivery": []
        }

    def log(self, delay, throughput, energy, delivery):
        self.data["delay"].append(delay)
        self.data["throughput"].append(throughput)
        self.data["energy"].append(energy)
        self.data["packet_delivery"].append(delivery)

    def summary(self):
        return {k: np.mean(v) if v else 0.0 for k, v in self.data.items()}
