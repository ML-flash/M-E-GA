import random
from M_E_GA_Base_V2 import M_E_GA_Base
from concurrent.futures import ThreadPoolExecutor
import os
from M_E_GA_fitness_funcs import LeadingOnesFitness  # Import the modified fitness function class

MAX_LENGTH = 300
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

def evaluate_population(population, encoding_manager, num_threads=None):
    if num_threads is None:
        num_threads = os.cpu_count()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda ind: evaluate_individual(ind, encoding_manager), population))
    return results

# Initialize the fitness function with update function passed in
fitness_function = LeadingOnesFitness(max_length=MAX_LENGTH, update_best_func=update_best_organism)
genes = fitness_function.genes

config = {
    'mutation_prob': 0.10,
    'delimited_mutation_prob': 0.05,
    'open_mutation_prob': 0.007,
    'capture_mutation_prob': 0.002,
    'delimiter_insert_prob': 0.05,
    'crossover_prob': 0.50,
    'elitism_ratio': 0.6,
    'base_gene_prob': 0.50,
    'capture_gene_prob': 0.1,
    'max_individual_length': 90,
    'population_size': 700,
    'num_parents': 150,
    'max_generations': 300,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': True,
    'generation_logging': False,
    'mutation_logging': True,
    'crossover_logging': False,
    'individual_logging': False,
    'seed': GLOBAL_SEED
}

# Initialize the GA with the selected genes and the fitness function's compute method
ga = M_E_GA_Base(genes, lambda ind, ga_instance: fitness_function.compute(ind, ga_instance), **config)

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