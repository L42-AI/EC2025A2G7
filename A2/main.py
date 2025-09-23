from EvolutionManager import EvolutionManager
from Controller import NNController
import numpy as np
import time as t

import run

if __name__ == "__main__":
    input_size = 29  # 15 qpos + 14 qvel
    hidden_size = 64
    output_size = 8  # 8 joints
    population_size = 1000
    generations = 10

    controller_type = NNController

    evolution_manager = EvolutionManager(
        input_size,
        hidden_size,
        output_size,
        controller_type=controller_type,
    )

    run.single(
        controller=controller_type(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            weights=None,
        ),
        simulation_steps=15_000,
        record_video=True,
    )

    population = evolution_manager.build_population(population_size)

    best_weights, logbook = evolution_manager.run_evolution(
        population.copy(),
        generations=generations,
        cx_prob=0.5,
        mut_prob=0.5
    )

    run.single(
        controller=controller_type(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            weights=np.array(best_weights),
        ),
        simulation_steps=15_000,
        record_video=True,
    )

    # best_weights, logbook = evolution_manager.run_evolution_infinite(
    #     population.copy(), cx_prob=0.5, mut_prob=0.5
    # )