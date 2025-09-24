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

    best_weights = np.load("ea_results/42_best_individual_curricular_learning_True.npy")

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