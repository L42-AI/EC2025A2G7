from experiment_runner import ExperimentRunner
from utils import read_in_all_csv_in_dir, read_in_all_npy_in_dir
from run_sim import run_multi_agent_world

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_random()