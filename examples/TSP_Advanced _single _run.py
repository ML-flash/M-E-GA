import random
from M_E_GA import M_E_GA_Base
from TSP_Advanced_FF import TspAdvanced

NUM_LOCATIONS = 1000
VALUE_RANGE = (30, 110)
COORD_RANGE = (0, 5000)


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


def evaluate_best_organism(fitness_function, best_organism, encoding_manager):
    print("\nEvaluating best organism with verbosity on...")
    fitness_score = fitness_function.compute(best_organism["genome"], encoding_manager, verbosity=True)
    print(f"Best organism fitness score with verbosity: {fitness_score}")

# Initialize the fitness function with update function passed in
fitness_function = TspAdvanced(NUM_LOCATIONS, value_range=VALUE_RANGE, coord_range=COORD_RANGE, update_best_func=update_best_organism)
genes = fitness_function.genes

config = {
    'mutation_prob': 0.05,
    'delimited_mutation_prob': 0.03,
    'open_mutation_prob': 0.007,
    'capture_mutation_prob': 0.003,
    'delimiter_insert_prob': 0.004,
    'crossover_prob': 0.70,
    'elitism_ratio': 0.4,
    'base_gene_prob': 0.45,
    'capture_gene_prob': 0.1,
    'max_individual_length': 100,
    'population_size': 700,
    'num_parents': 200,
    'max_generations': 1000,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': False,
    'generation_logging': False,
    'mutation_logging': False,
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


# Evaluate and display the best organism with verbosity on
evaluate_best_organism(fitness_function, best_organism, ga)