from EvolutionManager import EvolutionManager
from Controller import NNController
import numpy as np
import random as r

r.seed(24)
np.random.seed(24)

if __name__ == "__main__":
    input_size = 43  # 15 qpos + 14 qvel + 14 qacc
    hidden_size = 64
    output_size = 8  # 8 joints
    population_size = 200
    generations = 10

    controller_type = NNController

    evolution_manager = EvolutionManager(
        input_size,
        hidden_size,
        output_size,
        controller_type=controller_type
    )
    population = evolution_manager.build_population(population_size)

    fitnesses = evolution_manager.run_baseline(population)
    
    best_weights, logbook = evolution_manager.run_evolution(
        population.copy(),
        generations=generations,
        cx_prob=0.5,
        mut_prob=0.5
    )

    best_weights, logbook = evolution_manager.run_evolution(
        population.copy(),
        generations=generations,
        cx_prob=0.0,
        mut_prob=1.0,
        curricular_learning=True
    )
