import random
import time as t
from pathlib import Path
import multiprocessing as mp
from functools import partial

import numpy as np
from deap import base, creator, tools

from fitness_functions import (
    get_best_closeness_to_xyz,
    get_best_distance_from_start,
    get_target_fitness,
    get_highest_negative_y
)

import run
from Controller import NNController
from ctm_types import ControllerType
import evolution.crossover as cx
import evolution.algorithm as algorithms

mp.set_start_method("spawn", force=True)  # important on macOS

def evaluate_individual(
    individual: list[float],
    controller_type: type[ControllerType],
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> tuple[float, float]:

    controller = controller_type(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        weights=np.array(individual)
    )

    history = run.single(
        controller=controller,
        simulation_steps=100_000,
    )
    fitness, speed = get_highest_negative_y(history)
    return fitness, # Return a tuple of fitness and speed

class EvolutionManager:
    def __init__(self, input_size: int = 15, hidden_size: int = 64, output_size: int = 8, controller_type: type[ControllerType] = type[NNController], weights: list = None):
        brain = controller_type(input_size, hidden_size, output_size).brain
        num_weights = brain.get_num_weights()

        # Setup DEAP framework
        creator.create(
            "Fitness", base.Fitness, weights=(1.0,)
        ) # Maximize fitness, weights represents minimize (-1.0)/maximize(1.0)

        creator.create(
            "Individual", list, fitness=creator.Fitness
        ) # list is used to store weights in

        # Initialize toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -1.0, 1.0) # Weights between -1 and 1
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=num_weights)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2) # Gaussian mutation
        self.toolbox.register("select", tools.selTournament, tournsize=5) # Tournament selection, picking best of 5

        self.toolbox.register(
            "evaluate",
            evaluate_individual,
            controller_type=controller_type,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
        )


    # utility function for naming files
    @staticmethod
    def unique_stem(tag: str, rund_id: int) -> str:
        ts = t.strftime("%Y%m%d-%H%M%S")
        return f"{tag}_run{rund_id:02d}_{ts}"

    # convert scalar or list of probs into n_gen length numpy array of prob
    @staticmethod
    def to_series(prob: float | list | np.ndarray, n_gen: int):
        if isinstance(prob, (list, tuple, np.ndarray)):
            arr = np.array(prob, dtype=float)
            if len(arr) != n_gen:
                raise ValueError(
                    f"Length mismatch: got {len(arr)} rates, but {n_gen} generations."
                )
            return arr
        else:
            return np.full(n_gen, float(prob), dtype=float)

    @staticmethod
    # save generation-wise statistics from DEAP logbook to a .npz file
    def save_logbook(
        logbook: tools.Logbook,
        tag: str,
        run_id: int,
        out_dir=Path(__file__).parent / "results",
        mutpb: float | list | np.ndarray = None,
        cxpb: float | list | np.ndarray = None,
    ):

        gen = np.array(logbook.select("gen"))
        avg = np.array(logbook.select("avg"))
        std = np.array(logbook.select("std"))
        minv = np.array(logbook.select("min"))
        maxv = np.array(logbook.select("max"))

        # build unique filename
        stem = EvolutionManager.unique_stem(tag, run_id)
        npz_path = out_dir / f"{stem}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(npz_path, gen=gen, avg=avg, std=std, min=min, max=max)
        print(f"saved logbook in {npz_path}")

        n = len(gen)
        extra = {}
        if mutpb is not None:
            extra["mutpb"] = EvolutionManager.to_series(mutpb, n)
        if cxpb is not None:
            extra["cxpb"] = EvolutionManager.to_series(cxpb, n)

        np.savez_compressed(
            npz_path, gen=gen, avg=avg, std=std, min=minv, max=maxv, **extra
        )
        print(f"saved logbook in {npz_path}")

    def build_population(self, population_size: int) -> list:
        print("Starting evolution with population size:", population_size)
        return self.toolbox.population(n=population_size)

    def run_evolution(
        self,
        pop: list,
        generations: int = 20,
        cx_prob: float = 0.8,
        mut_prob: float = 0.3,
        curricular_learning: bool = False,

    ) -> tuple[np.ndarray, tools.Logbook]:

        if curricular_learning:
            cx_schedule = partial(cx.get_linear_cx_prob_schedule, start_prob=cx_prob, end_prob=1 - cx_prob)
            mut_schedule = partial(cx.get_linear_mut_prob_schedule, start_prob=mut_prob, end_prob=1 - mut_prob)
        else:
            cx_schedule = None
            mut_schedule = None

        # Track best fitness
        hof = tools.HallOfFame(1)

        # Track stats across generations
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        start = t.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(mp.cpu_count()) as pool:
            self.toolbox.register("map", pool.map)

            pop, logbook = algorithms.eaMuPlusLambda(
                pop,
                self.toolbox,
                mu=len(pop),
                lambda_=len(pop),
                cxpb=cx_prob,
                mutpb=mut_prob,
                ngen=generations,
                stats=stats,
                halloffame=hof,
                cx_schedule=cx_schedule,
                mut_schedule=mut_schedule,
                verbose=True
            )
    
        end = t.time()
        print(f"Time taken: {end - start:.2f} seconds")

        self.save_logbook(
            logbook,
            tag="EA1",
            run_id=1,
            out_dir=Path(__file__).parent / "results",
            mutpb=mut_prob,
            cxpb=cx_prob,
        )

        return np.array(hof[0]), logbook
