from functools import partial
import multiprocessing as mp
import random
import time as t
from pathlib import Path
mp.set_start_method("spawn", force=True)  # important on macOS

import numpy as np
from deap import base, creator, tools, algorithms

from fitness_functions import get_best_closeness_to_xyz, get_best_distance_from_start, get_target_fitness
from Controller import NNController
import run

# fitness_func = get_best_distance_from_start
fitness_func = partial(get_best_closeness_to_xyz, target=np.array([10.0, 0.0, 0.0]))

def evaluate_individual(individual, input_size: int, hidden_size: int, output_size: int) -> tuple:
    controller = NNController(
        input_size=input_size, 
        hidden_size=hidden_size, 
        output_size=output_size, 
        weights=np.array(individual)
    )

    history = run.single(
        controller=controller,
        simulation_steps=10_000,
    )
    fitness = fitness_func(history)
    return (fitness,) # Return a tuple of fitness
    
class EvolutionManager:



    def __init__(self, input_size: int = 15, hidden_size: int = 64, output_size: int = 8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.num_weights =  (
            (input_size * hidden_size + hidden_size)
            + (hidden_size * hidden_size + hidden_size)
            + (hidden_size * output_size + output_size)
        )

        # Setup DEAP framework
        creator.create("FitnessMin", base.Fitness, weights=(1.0,)) # Maximize fitness, weights represents minimize (-1.0)/maximize(1.0)
        creator.create("Individual", list, fitness=creator.FitnessMin) # list is used to store weights in

        # Initialize toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -1.0, 1.0) # Weights between -1 and 1
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.num_weights)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5) # Uniform crossover, keeping individual length constant
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2) # Gaussian mutation
        self.toolbox.register("select", tools.selTournament, tournsize=5) # Tournament selection, picking best of 5

        self.toolbox.register("evaluate", evaluate_individual, input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    @staticmethod
    #save generation-wise statistics from DEAP logbook to a .npz file     
    def save_logbook(logbook, tag:str, run_id:int, out_dir="logbook_results"):
        gen = np.array(logbook.select("gen"))
        avg = np.array(logbook.select("avg"))
        std = np.array(logbook.select("std"))
        min = np.array(logbook.select("min"))
        max = np.array(logbook.select("max"))

        out = Path(out_dir) / f"{tag}_run{run_id:02d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)

        np.savez(out, gen=gen, avg=avg, std=std, min=min, max=max)
        print("saved logbook in {out}")
        return None   
    
    def run_evolution(self, population_size=200, generations=20, cx_prob=0.8, mut_prob=0.3, multi: bool=True):
        print("Starting evolution with population size:", population_size)
        pop = self.toolbox.population(n=population_size)

        # Track best fitness
        hof = tools.HallOfFame(1)

        # Track stats across generations
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        start = t.time()
        if multi:
            # --- Use multiprocessing only inside a context ---
            print("Using multiprocessing, with", mp.cpu_count(), "cores")
            ctx = mp.get_context("spawn")
            with ctx.Pool(mp.cpu_count()) as pool:
                self.toolbox.register("map", pool.map)

                pop, logbook = algorithms.eaMuPlusLambda(
                    pop, self.toolbox,
                    mu=population_size,
                    lambda_=population_size,
                    cxpb=cx_prob,
                    mutpb=mut_prob,
                    ngen=generations,
                    stats=stats,
                    halloffame=hof,
                    verbose=True
                )
        else:
            pop, logbook = algorithms.eaMuPlusLambda(
                pop, self.toolbox,
                mu=population_size,
                lambda_=population_size,
                cxpb=cx_prob,
                mutpb=mut_prob,
                ngen=generations,
                stats=stats,
                halloffame=hof,
                verbose=True
            )

        end = t.time()
        print(f"Time taken: {end - start:.2f} seconds")

        # Extract best fitness history
        gen = logbook.select("gen")
        min_fitness = logbook.select("min")

        best_ind = hof[0]

        # Print progression
        print("Fitness progression:")
        for g, f in zip(gen, min_fitness):
            print(f"Gen {g}: Best Fitness = {f}")

        # print("Best individual is:", best_ind, "Fitness:", best_ind.fitness.values)

        self.save_logbook(logbook, tag="EA1", run_id=1, out_dir="A2/results")


        return hof[0], logbook
    
    
   
    
        
        
        
