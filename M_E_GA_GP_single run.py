import random
from M_E_GA_Base_V2 import M_E_GA_Base
from concurrent.futures import ThreadPoolExecutor
import os
from M_E_GA_GP_interpreter import MegaGP  # Assuming MegaGP is correctly imported
from M_E_GA_GP_fitness_template import MegaGPFitnessFunction
from itertools import product
import math


INPUT_SIZE = 6  # Set based on the expected number of input bits for MegaGP
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)

best_organism = {
    "genome": None,
    "fitness": float('-inf')
}

def update_best_organism(current_genome, current_fitness, verbose=False):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")

def evaluate_population(population, encoding_manager, num_threads=16):
    if num_threads is None:
        num_threads = os.cpu_count()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda ind: evaluate_individual(ind, encoding_manager), population))
    return results

# Initialize MegaGP and MegaGPFitnessFunction
fitness_function = MegaGPFitnessFunction(INPUT_SIZE, update_best_func=update_best_organism)

config = {
    'mutation_prob': 0.15,
    'delimited_mutation_prob': 0.03,
    'open_mutation_prob': 0.0005,
    'capture_mutation_prob': 0.0005,
    'delimiter_insert_prob': 0.002,
    'crossover_prob': 0.50,
    'elitism_ratio': 0.6,
    'base_gene_prob': 0.45,
    'capture_gene_prob': 0.15,
    'max_individual_length': 40,
    'population_size': 700,
    'num_parents': 150,
    'max_generations': 300,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': False,
    'generation_logging': False,
    'mutation_logging': True,
    'crossover_logging': False,
    'individual_logging': False,
    'seed': GLOBAL_SEED
}

# Initialize the GA with the selected genes from MegaGPFitnessFunction and the fitness function's compute method
ga = M_E_GA_Base(fitness_function.genes, lambda ind, ga_instance: fitness_function.compute(ind, ga_instance), **config)

# Run the GA
ga.run_algorithm()

# Find the best solution
best_genome = best_organism["genome"]
best_fitness = best_organism["fitness"]
best_solution_decoded = ga.decode_organism(best_genome, format=True)

print('Length of best solution:', len(best_solution_decoded))
print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
print('Length of best genome:', len(best_organism["genome"]))
print(f"Best Genome (Encoded): {best_genome}")

# Calculate on the best genome with verbosity
best = fitness_function.compute(best_genome, ga, verbose=True)

print(f"Best fitness score: {best}")
