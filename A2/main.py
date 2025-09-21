from EvolutionManager import EvolutionManager
from Controller import NNController
import numpy as np
import time as t

import run

if __name__ == "__main__":
    input_size = 15  # 15 qpos 
    hidden_size = 32
    output_size = 8  # 8 joints
    population_size = 200
    generations = 30

    controller_type = NNController

    evolution_manager = EvolutionManager(
        input_size,
        hidden_size,
        output_size,
        controller_type=controller_type
    )
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
    # best_weights, logbook = evolution_manager.run_evolution(
    #     population_size=population_size,
    #     generations=generations,
    #     cx_prob=0.5,
    #     mut_prob=0.5
    # )
    best_weights, logbook = evolution_manager.run_evolution(
        population_size=population_size,
        generations=generations,
        cx_prob=0.0,
        mut_prob=1.0,
        curricular_learning=True
    )