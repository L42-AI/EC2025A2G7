from EvolutionManager import EvolutionManager
from Controller import NNController, RandomController
import numpy as np
import run

if __name__ == "__main__":
    # ExperimentRunner()._run_experiment(controller=NNController(input_size=15, hidden_size=64, output_size=8), record_video=True)
    evolution_manager = EvolutionManager(input_size=15, hidden_size=64, output_size=8)
    best_individual, logbook = evolution_manager.run_evolution(population_size=50, generations=10, cx_prob=0.7, mut_prob=0.3)
    run.single(controller=NNController(input_size=15, hidden_size=64, output_size=8, weights=np.array(best_individual)), record_video=True)