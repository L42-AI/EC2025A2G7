from deap import base, creator, tools, algorithms
import random
import numpy as np
from fitness_functions import get_furthest_distance
from evolutionNN import NNController
from experiment_runner import ExperimentRunner

class EvolutionManager:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.num_weights = (input_size * hidden_size) + (hidden_size * output_size)
        self.evaluate_fitness = get_furthest_distance
        
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
        experiment = ExperimentRunner()
        controller = NNController(input_size=self.input_size, 
                                  hidden_size=self.hidden_size, 
                                  output_size=self.output_size, 
                                  weights=np.array(individual))
        result = experiment._run_experiment(
            controller=controller,
            simulation_steps=500_000,
        )
        fitness = self.evaluate_fitness(result)
        return fitness
    
    def run_evolution(self, population_size=50, generations=20, cx_prob=0.5, mut_prob=0.2):
        pop = self.toolbox.population(population_size)
        print("Starting evolution with population size:", population_size)
        best_ind = algorithms.eaSimple(pop, self.toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=generations, verbose=True)
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is:", best_ind)
        return best_ind
            
    
        
        
        
