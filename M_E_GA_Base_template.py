import random
from M_E_GA_Base_V2 import M_E_GA_Base

MAX_LENGTH = 100
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

# Bulk fitness function to process the entire population.
def evaluate_population(population, encoding_manager):
    population_fitness = []
    for individual in population:
        fitness_score = leading_ones_fitness_function(individual, encoding_manager)
        population_fitness.append(fitness_score)
        update_best_organism(encoded_individual, fitness_score, verbose=True)
    return population_fitness

# Individual fitness function.
def leading_ones_fitness_function(encoded_individual, ga_instance):
    decoded_individual = ga_instance.decode_organism(encoded_individual)
    fitness_score = sum(1 ** i if gene == '1' else 0 for i, gene in enumerate(decoded_individual))
    penalty = (1.05 ** (MAX_LENGTH - len(decoded_individual)) if len(decoded_individual) < MAX_LENGTH else (
                len(decoded_individual) - MAX_LENGTH))
    update_best_organism(encoded_individual, fitness_score, verbose=True)
    return fitness_score - penalty

genes = ['0', '1']

config = {
    'mutation_prob': 0.01,
    'delimited_mutation_prob': 0.01,
    'open_mutation_prob': 0.007,
    'capture_mutation_prob': 0.002,
    'delimiter_insert_prob': 0.004,
    'crossover_prob': 0.45,
    'elitism_ratio': 0.5,
    'base_gene_prob': 0.50,
    'max_individual_length': 40,
    'population_size': 500,
    'num_parents': 100,
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
