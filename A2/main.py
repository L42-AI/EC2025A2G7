from EvolutionManager import EvolutionManager
from Controller import NNController
import numpy as np
import time as t

import run

if __name__ == "__main__":
    input_size = 29  # 15 qpos + 14 qvel
    hidden_size = 64
    output_size = 8  # 8 joints
    population_size = 500
    generations = 100

    controller_type = NNController

    evolution_manager = EvolutionManager(
        input_size,
        hidden_size,
        output_size,
        controller_type=controller_type,
    )

    population = evolution_manager.build_population_from_file('best_individual.npy', population_size)

    best_weights, logbook = evolution_manager.run_evolution_infinite(
        population.copy(), cx_prob=0.5, mut_prob=0.5
    )