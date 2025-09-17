from experiment_runner import ExperimentRunner
from utils import read_in_all_csv_in_dir, read_in_all_npy_in_dir
from run_sim import run_multi_agent_world
from EvolutionManager import EvolutionManager
from evolutionNN import NNController
import numpy as np

if __name__ == "__main__":
    runner = ExperimentRunner()
    # runner.run_random()
    evolution_manager = EvolutionManager(input_size=15, hidden_size=8, output_size=8)
    best_individual = evolution_manager.run_evolution() 
    runner._run_experiment(controller=NNController(weights=np.array(best_individual)), visualise=True, simulation_steps=2_000_000)