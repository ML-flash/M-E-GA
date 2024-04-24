import random

# Assuming M_E_GA_Base and other necessary modules are properly defined and imported
from M_E_GA_Base_V2 import M_E_GA_Base

MAX_LENGTH = 100

GLOBAL_SEED =  None
random.seed(GLOBAL_SEED)  # Ensure reproducibility for research purposes.


best_organism = {
    "genome": None,
    "fitness": float('-inf')  # Start with negative infinity to ensure any valid organism will surpass it
}

def update_best_organism(current_genome, current_fitness, verbose = False):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")

# Step 1: Define the Fitness Function
def leading_ones_fitness_function(encoded_individual, encoding_manager):
    penalty = 0
    # Decode the individual
    decoded_individual = encoding_manager.decode_organism(encoded_individual)


    fitness_score = 0
    counting = True  # Start counting leading ones
    ones = 0

    for gene in decoded_individual:
        if gene == 'Start' or gene == 'End':
            continue  # Skip delimiters
        if gene == '1' and counting:
            ones += 1
        elif gene == '0':
            break  # Stop counting after the first '0' and exit the loop

    if len(decoded_individual) < MAX_LENGTH:
        pass
        penalty +=  1.05 ** (MAX_LENGTH - len(decoded_individual))

    if len(decoded_individual) > MAX_LENGTH:
        penalty += (len(decoded_individual) - MAX_LENGTH)
    fitness_score += 1.05 ** ones
    fitness_score -= penalty

    update_best_organism(encoded_individual, fitness_score, verbose=True)

    return fitness_score



genes = ['0', '1',]


config = {
    'mutation_prob': .50,
    'delimited_mutation_prob': 0.40,
    'open_mutation_prob': 0.0008,
    'capture_mutation_prob': 0.001,
    'delimiter_insert_prob': 0.0008,
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


ga = M_E_GA_Base(
    genes=genes,
    fitness_function=leading_ones_fitness_function,
    **config
)

ga.run_algorithm()

best_genome = best_organism["genome"]
best_fitness = best_organism["fitness"]
best_solution_decoded = ga.decode_organism(best_genome, format=True)
print('length', len(best_solution_decoded))
print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
print('length',len(best_organism["genome"]))
print(f"Best Genome (Encoded): {best_genome}")
