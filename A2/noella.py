from EvolutionManager import EvolutionManager
from Controller import NNController
import numpy as np
import random as r

from consts import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, POPULATION_SIZE, GENERATIONS

r.seed(98)
np.random.seed(98)

if __name__ == "__main__":
    controller_type = NNController

    evolution_manager = EvolutionManager(
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE,
        controller_type=controller_type,
    )
    population = evolution_manager.build_population(POPULATION_SIZE)

    fitnesses = evolution_manager.run_baseline(population)

    best_weights, logbook = evolution_manager.run_evolution(
        population.copy(), generations=GENERATIONS, cx_prob=0.5, mut_prob=0.5
    )

    best_weights, logbook = evolution_manager.run_evolution(
        population.copy(),
        generations=GENERATIONS,
        cx_prob=0.0,
        mut_prob=1.0,
        curricular_learning=True,
    )
