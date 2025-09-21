import random
import numpy as np

def get_linear_cx_prob_schedule(ngens, gen, start_prob=0.1, end_prob=0.9):
    """Returns a function that computes a linearly decreasing crossover probability."""
    return start_prob + (end_prob - start_prob) * (gen / ngens)

def get_exponential_cx_prob_schedule(ngens, gen, start_prob=0.1, end_prob=0.9):
    """Returns a function that computes an exponentially decreasing crossover probability."""
    return start_prob * (end_prob / start_prob) ** (gen / ngens)

def get_logistic_cx_prob_schedule(ngens, gen, start_prob=0.1, end_prob=0.9, steepness=10):
    """Returns a function that computes a logistic crossover probability."""
    midpoint = ngens / 2
    return start_prob + (end_prob - start_prob) / (1 + np.exp(-steepness * (gen - midpoint) / ngens))

def get_linear_mut_prob_schedule(ngens, gen, start_prob=0.9, end_prob=0.1):
    """Returns a function that computes a linearly decreasing mutation probability."""
    return start_prob + (end_prob - start_prob) * (gen / ngens)

def get_exponential_mut_prob_schedule(ngens, gen, start_prob=0.9, end_prob=0.1):
    """Returns a function that computes an exponentially decreasing mutation probability."""
    return start_prob * (end_prob / start_prob) ** (gen / ngens)

def get_logistic_mut_prob_schedule(ngens, gen, start_prob=0.9, end_prob=0.1, steepness=10):
    """Returns a function that computes a logistic mutation probability."""
    midpoint = ngens / 2
    return start_prob + (end_prob - start_prob) / (1 + np.exp(-steepness * (gen - midpoint) / ngens))

import random
import numpy as np
from deap import base, creator, tools

# Helper to get indices of weight blocks
def get_submodule_indices(modular_brain):
    """
    Returns a dict mapping submodule/interconnector name -> slice indices in the flat weight vector
    """
    indices = {}
    offset = 0

    # Submodules first
    for name, dims in modular_brain.submodules.items():
        if name == "global":
            W1_size = dims["in"] * modular_brain.hidden_size
            W2_size = modular_brain.hidden_size * dims["out"]
        else:
            W1_size = (dims["in"] + modular_brain.hidden_size) * modular_brain.hidden_size
            W2_size = modular_brain.hidden_size * dims["out"]
        total = W1_size + W2_size
        indices[name] = slice(offset, offset + total)
        offset += total

    # Interconnectors
    for name, submods in modular_brain.interconnectors.items():
        in_size = sum([modular_brain.submodules[s]["out"] for s in submods])
        out_size = in_size
        W1_size = in_size * modular_brain.hidden_size
        W2_size = modular_brain.hidden_size * out_size
        total = W1_size + W2_size
        indices[name] = slice(offset, offset + total)
        offset += total

    return indices


# ---------------------
# Crossover: swap full submodule weights
# ---------------------
def modular_crossover(ind1, ind2, indices, cx_prob=0.5):
    """
    ind1, ind2: DEAP individuals (flat weight arrays)
    modular_brain: instance of ModularNNBrain (to know block indices)
    """
    # Pick 1 or more submodules/interconnectors to swap
    names_to_swap = random.sample(list(indices.keys()), k=random.randint(1, len(indices)))
    for name in names_to_swap:
        if random.random() > cx_prob:
            continue  # Skip based on crossover probability
        sl = indices[name]
        ind1[sl], ind2[sl] = ind2[sl].copy(), ind1[sl].copy()  # swap weights
    return ind1, ind2


# ---------------------
# Mutation: mutate a single submodule/interconnector
# ---------------------
def modular_mutation(individual, indices, mu=0.0, sigma=0.3, indpb=0.2):
    # Pick a random submodule

    name = random.choice(list(indices.keys()))
    sl = indices[name]

    # Extract block as NumPy array
    block = np.array(individual[sl])

    # Apply mutation
    mutation_mask = np.random.rand(block.size) < indpb
    noise = np.random.normal(mu, sigma, size=block.size)
    block[mutation_mask] += noise[mutation_mask]

    # Write back into individual (as list)
    individual[sl] = block.tolist()
    return individual,


# ---------------------
# Example: register with DEAP toolbox
# ---------------------
# Assuming `brain` is an instance of ModularNNBrain
# toolbox = base.Toolbox()
# toolbox.register("mate", modular_crossover, modular_brain=brain)
# toolbox.register("mutate", modular_mutation, modular_brain=brain, mu=0.0, sigma=0.2, indpb=0.1)
