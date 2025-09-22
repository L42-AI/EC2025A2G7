from EvolutionManager import EvolutionManager
from Controller import NNController
import numpy as np
import time as t

import run

if __name__ == "__main__":
    input_size = 43  # 15 qpos + 14 qvel + 14 qacc
    hidden_size = 64
    output_size = 8  # 8 joints
    population_size = 10
    generations = 10

    controller_type = NNController

    evolution_manager = EvolutionManager(
        input_size,
        hidden_size,
        output_size,
        controller_type=controller_type,
        exp_name=None    )
    population = evolution_manager.build_population(population_size)

    fitnesses = evolution_manager.run_baseline(population)
    

    # best_weights, logbook = evolution_manager.run_evolution(
    #     population_size=population_size,
    #     generations=generations,
    #     cx_prob=0.0,
    #     mut_prob=1.0
    # )
    # best_weights, logbook = evolution_manager.run_evolution(
    #     population_size=population_size,
    #     generations=generations,
    #     cx_prob=1.0,
    #     mut_prob=0.0
    # )

    # run.single(controller_type(input_size, hidden_size, output_size, weights=None), record_video=True)


    best_weights, logbook = evolution_manager.run_evolution(
        population.copy(),
        generations=generations,
        cx_prob=0.5,
        mut_prob=0.5
    )
    run.single(controller_type(input_size, hidden_size, output_size, weights=best_weights), record_video=True)

    best_weights, logbook = evolution_manager.run_evolution(
        population.copy(),
        generations=generations,
        cx_prob=0.0,
        mut_prob=1.0,
        curricular_learning=True
    )
    run.single(controller_type(input_size, hidden_size, output_size, weights=best_weights), record_video=True)
