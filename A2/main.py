from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from experiment_runner import ExperimentRunner
from utils import read_in_all_csv_in_dir, read_in_all_npy_in_dir
from EvolutionManager import EvolutionManager
from Controller import NNController, RandomController
import numpy as np
from run_sim import main
from ctm_types import WORLD_MAP

if __name__ == "__main__":
    ExperimentRunner()._run_experiment(controller=NNController(input_size=15, hidden_size=64, output_size=8), record_video=True)
    evolution_manager = EvolutionManager(input_size=15, hidden_size=64, output_size=8)
    best_individual, logbook = evolution_manager.run_evolution(population_size=100, generations=20, cx_prob=0.7, mut_prob=0.3)
    ExperimentRunner()._run_experiment(controller=NNController(input_size=15, hidden_size=64, output_size=8, weights=np.array(best_individual)), record_video=True)