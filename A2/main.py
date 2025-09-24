from EvolutionManager import EvolutionManager
from Controller import NNController
import numpy as np
import time as t

import run

from consts import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, POPULATION_SIZE, GENERATIONS

if __name__ == "__main__":

    controller_type = NNController

    evolution_manager = EvolutionManager(
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE,
        controller_type=controller_type,
    )

    best_weights = np.load("results/best_individual_curricular_learning_False.npy")

    run.single(
        controller=controller_type(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            weights=np.array(best_weights),
        ),
        simulation_steps=15_000,
        record_video=True,
    )

    population = evolution_manager.build_population(POPULATION_SIZE)

    best_weights, logbook = evolution_manager.run_evolution(
        population.copy(),
        generations=GENERATIONS,
        cx_prob=0.5,
        mut_prob=0.5
    )

    run.single(
        controller=controller_type(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            weights=np.array(best_weights),
        ),
        simulation_steps=15_000,
        record_video=True,
    )

    # best_weights, logbook = evolution_manager.run_evolution_infinite(
    #     population.copy(), cx_prob=0.5, mut_prob=0.5
    # )