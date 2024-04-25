import random
from M_E_GA_Base_V2 import M_E_GA_Base
from concurrent.futures import ThreadPoolExecutor
import os


MAX_LENGTH = 30000
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)

# Placeholder for the best organism found during the GA run.
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

from concurrent.futures import ThreadPoolExecutor


def evaluate_population(population, encoding_manager, num_threads=None):
    # If no specific number of threads is provided, use the number of CPUs available
    if num_threads is None:
        num_threads = os.cpu_count()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Map each individual in the population to the evaluate_individual function
        results = list(executor.map(lambda ind: evaluate_individual(ind, encoding_manager), population))
    return results

# Individual fitness function.
def leading_ones_fitness_function(encoded_individual, ga_instance):
    decoded_individual = ga_instance.decode_organism(encoded_individual)
    fitness_score = sum(1 ** i if gene == '1' else 0 for i, gene in enumerate(decoded_individual))
    penalty = (1.008 ** (MAX_LENGTH - len(decoded_individual)) if len(decoded_individual) < MAX_LENGTH else (
                len(decoded_individual) - MAX_LENGTH))
    update_best_organism(encoded_individual, fitness_score, verbose=True)
    return fitness_score - penalty

genes = ['0', '1']

config = {
    'mutation_prob': 0.02,
    'delimited_mutation_prob': 0.01,
    'open_mutation_prob': 0.007,
    'capture_mutation_prob': 0.002,
    'delimiter_insert_prob': 0.004,
    'crossover_prob': 0.40,
    'elitism_ratio': 0.6,
    'base_gene_prob': 0.55,
    'capture_gene_prob': 0.15,
    'max_individual_length': 25,
    'population_size': 700,
    'num_parents': 150,
    'max_generations': 800,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': False,
    'generation_logging': True,
    'mutation_logging': False,
    'crossover_logging': False,
    'individual_logging': False,
    'seed': GLOBAL_SEED
}

# Initialize the GA
ga = M_E_GA_Base(genes, leading_ones_fitness_function, **config)

# Run the GA
ga.run_algorithm()

# Evaluate the entire population after the GA run
final_population = ga.get_population()  # Assuming a method to retrieve the population
final_population_fitness = evaluate_population(final_population, ga.encoding_manager)

# Find the best solution
best_genome = best_organism["genome"]
best_fitness = best_organism["fitness"]
best_solution_decoded = ga.decode_organism(best_genome, format=True)

print('Length of best solution:', len(best_solution_decoded))
print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
print('Length of best genome:', len(best_organism["genome"]))
print(f"Best Genome (Encoded): {best_genome}")
