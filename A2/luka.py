from EvolutionManager import EvolutionManager
from Controller import NNController
import numpy as np
import random as r

from consts import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, POPULATION_SIZE, GENERATIONS

if __name__ == "__main__":

    seed = 42
    r.seed(seed)
    np.random.seed(seed)
    
    controller_type = NNController

    evolution_manager = EvolutionManager(
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE,
        controller_type=controller_type
    )

    population = evolution_manager.build_population(POPULATION_SIZE)
    
    for gen in range(GENERATIONS):
        fitnesses = evolution_manager.run_baseline(population, seed, gen)

    seed = 123
    r.seed(seed)
    np.random.seed(seed)
    
    controller_type = NNController

    evolution_manager = EvolutionManager(
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE,
        controller_type=controller_type
    )

    population = evolution_manager.build_population(POPULATION_SIZE)
    
    for gen in range(GENERATIONS):
        fitnesses = evolution_manager.run_baseline(population, seed, gen)

    seed = 13
    r.seed(seed)
    np.random.seed(seed)
    
    controller_type = NNController

    evolution_manager = EvolutionManager(
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE,
        controller_type=controller_type
    )

    population = evolution_manager.build_population(POPULATION_SIZE)
    
    for gen in range(GENERATIONS):
        fitnesses = evolution_manager.run_baseline(population, seed, gen)

    seed = 24
    r.seed(seed)
    np.random.seed(seed)
    
    controller_type = NNController

    evolution_manager = EvolutionManager(
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE,
        controller_type=controller_type
    )

    population = evolution_manager.build_population(POPULATION_SIZE)
    
    for gen in range(GENERATIONS):
        fitnesses = evolution_manager.run_baseline(population, seed, gen)