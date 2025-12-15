import random
from M_E_GA_Base import M_E_GA_Base
from concurrent.futures import ThreadPoolExecutor
import os
from I_Dont_Know_Drop import IDontKnow

GLOBAL_SEED = None
random.seed(GLOBAL_SEED)


VOLUME = 15
NUM_ITEMS = 500
NUM_GROUPS = 6
MAX_SIZE = 500
MAX_WEIGHT = 20
MAX_DENSITY = 200


GLOBAL_SEED = None
random.seed(GLOBAL_SEED)

best_organism = {
    "genome": None,
    "fitness": float('-inf')
}

def update_best_organism(current_genome, current_fitness, verbose=True):
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
fitness_function = IDontKnow(
    volume=VOLUME,
    num_items=NUM_ITEMS,
    num_groups=NUM_GROUPS,
    update_best_func=update_best_organism,
    max_size=MAX_SIZE,
    max_weight=MAX_WEIGHT,
    max_density=MAX_DENSITY
)
genes = fitness_function.genes

config = {
    'mutation_prob': 0.15,
    'delimited_mutation_prob': 0.10,
    'open_mutation_prob': 0.06,
    'capture_mutation_prob': 0.08,
    'delimiter_insert_prob': 0.05,
    'delimit_delete_prob': 0.1,
    'crossover_prob': 0.0,
    'elitism_ratio': 0.00,
    'base_gene_prob': 0.30,
    'capture_gene_prob': 0.05,
    'max_individual_length': 200,
    'population_size': 500,
    'num_parents': 200,
    'max_generations': 1000,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': True,
    'generation_logging': True,
    'mutation_logging': True,
    'crossover_logging': True,
    'individual_logging': True,
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
