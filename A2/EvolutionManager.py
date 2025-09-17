from deap import base, creator, tools
import random
import numpy as np
import mujoco
from fitness_functions import xy_displacement
from evolutionNN import NNController

class EvolutionManager:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.num_weights = (input_size * hidden_size) + (hidden_size * output_size)
        self.evaluate_fitness = xy_displacement  # Example fitness function
        
        # Setup DEAP framework
        creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize fitness, weights represents minimize (-1.0)/maximize(1.0)
        creator.create("Individual", list, fitness=creator.FitnessMax) # list is used to store weights in
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -1.0, 1.0) # Weights between -1 and 1
        # Create an individual, which is a list of weights
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.num_weights)
        # Create a population of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint) # Two-point crossover, keeping individual length constant
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpd=0.1) # Gaussian mutation
        self.toolbox.register("select", tools.selTournament, tournsize=5) # Tournament selection, picking best of 5
        
    def evaluate_individual(self, individual):
        controller = NNController(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            weights=individual
        )
        # Evaluate the controller's performance
        return 
        
        
