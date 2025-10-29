import yaml
from src.simulation import run_simulation
from src.visualize import plot_metrics

if __name__ == "__main__":
    with open("config/example_config.yaml") as f:
        config = yaml.safe_load(f)

    metrics = run_simulation(config)
    plot_metrics(metrics)
