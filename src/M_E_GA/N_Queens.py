# experiment_runner.py
import random
from M_E_GA_Base import M_E_GA_Base  # Adjust the import if needed
from M_E_GA_fitness_funcs import NQueensFitness  # Import the NQueens fitness function
import datetime

# -------------------------------
# Global Settings and Seed
# -------------------------------
N = 8  # Board size (for 8-queens, for example)
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)

# -------------------------------
# Best Organism Callback
# -------------------------------
best_organism = {
    "genome": None,
    "fitness": float('-inf'),
    "complete_count": 0,
    "row_count": 0  # New: track the number of rows in the solution.
}

def update_best_organism(current_genome, current_fitness, complete_count=0, row_count=0, verbose=False):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        best_organism["complete_count"] = complete_count
        best_organism["row_count"] = row_count
        if verbose:
            print(f"New best organism found with fitness {current_fitness}, {complete_count} complete boards, and {row_count} rows.")

# -------------------------------
# Initialize Fitness Function
# -------------------------------
fitness_function = NQueensFitness(n=N, update_best_func=update_best_organism)
genes = fitness_function.genes

# -------------------------------
# GA Configuration Parameters
# -------------------------------
config = {
    'mutation_prob': 0.10,
    'delimited_mutation_prob': 0.05,
    'open_mutation_prob': 0.06,
    'metagene_mutation_prob': 0.10,  # Parameter for capturing metagenes
    'delimiter_insert_prob': 0.04,
    'crossover_prob': 0.0,
    'elitism_ratio': 0.7,
    'base_gene_prob': 0.44,
    'metagene_prob': 0.04,           # Controls meta gene selection probability
    # Let each candidate represent one or more board states.
    'max_individual_length': N * 3,  # For example, 3 boards per candidate
    'population_size': 700,
    'num_parents': 200,
    'max_generations': 500,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': False,
    'generation_logging': False,
    'mutation_logging': False,
    'crossover_logging': False,
    'individual_logging': False,
    'seed': GLOBAL_SEED,
    'lru_cache_size': 200  # LRU cache size for the EncodingManager
}

# -------------------------------
# Initialize and Run the GA
# -------------------------------
# We pass the fitness function's compute method (wrapped in a lambda) to the GA.
ga = M_E_GA_Base(genes, lambda ind, ga_instance: fitness_function.compute(ind, ga_instance), **config)

ga.run_algorithm()

# -------------------------------
# Print the Best Solution
# -------------------------------
best_genome = best_organism["genome"]
best_fitness = best_organism["fitness"]
complete_count = best_organism["complete_count"]
row_count = best_organism["row_count"]
best_solution_decoded = ga.decode_organism(best_genome, format=True)

print('Length of best solution (decoded):', len(best_solution_decoded))
print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
print(f"Number of complete boards in best solution: {complete_count}")
print(f"Total row count in best solution: {row_count}")
print('Length of best genome (encoded):', len(best_genome))
print(f"Best Genome (Encoded): {best_genome}")
