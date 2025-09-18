from functools import partial
import multiprocessing as mp
import random

import numpy as np
from deap import base, creator, tools, algorithms

from fitness_functions import get_furthest_xyz_distance
from evolutionNN import NNController
from experiment_runner import ExperimentRunner

class EvolutionManager:
    def __init__(self, input_size: int = 15, hidden_size: int = 64, output_size: int = 8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.num_weights =  (
            (input_size * hidden_size)
            + (hidden_size * hidden_size)
            + (hidden_size * output_size)
        )

        self.evaluate_fitness = partial(get_furthest_xyz_distance, target=np.array([0.0, -10.0, 0.0]))

        # Setup DEAP framework
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Maximize fitness, weights represents minimize (-1.0)/maximize(1.0)
        creator.create("Individual", list, fitness=creator.FitnessMin) # list is used to store weights in

        # Initialize toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -1.0, 1.0) # Weights between -1 and 1
        # Create an individual, which is a list of weights
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.num_weights)
        # Create a population of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evaluate_individual)

        self.toolbox.register("mate", tools.cxTwoPoint) # Two-point crossover, keeping individual length constant
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1) # Gaussian mutation
        self.toolbox.register("select", tools.selTournament, tournsize=5) # Tournament selection, picking best of 5
        
    def evaluate_individual(self, individual):
        experiment = ExperimentRunner()
        controller = NNController(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            output_size=self.output_size, 
            weights=np.array(individual)
        )

        result = experiment._run_experiment(
            controller=controller,
            simulation_steps=6000,
        )
        fitness = self.evaluate_fitness(result)
        return (fitness,) # Return a tuple of fitness
    
    def run_evolution(self, population_size=200, generations=20, cx_prob=0.8, mut_prob=0.3):
        print("Starting evolution with population size:", population_size)

        # Track best fitness
        hof = tools.HallOfFame(1)

        # Track stats across generations
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop = self.toolbox.population(population_size)
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox,
            cxpb=0.8, mutpb=0.3,
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        # Extract best fitness history
        gen = logbook.select("gen")
        min_fitness = logbook.select("min")

        # Print progression
        print("Fitness progression:")
        for g, f in zip(gen, min_fitness):
            print(f"Gen {g}: Best Fitness = {f}")

        # print("Best individual is:", best_ind, "Fitness:", best_ind.fitness.values)

        return hof[0], logbook
            
    
        
        
        
