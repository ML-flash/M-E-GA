import random
from M_E_GA_Base_V2 import M_E_GA_Base
from concurrent.futures import ThreadPoolExecutor
import os

MAX_LENGTH = 100
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)

def evaluate_population(population, encoding_manager, num_threads=32):
    if num_threads is None:
        num_threads = os.cpu_count()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda ind: evaluate_individual(ind, encoding_manager), population))
    return results


def leading_ones_fitness_function(encoded_individual, ga_instance):
    # Decode the individual
    decoded_individual = ga_instance.decode_organism(encoded_individual)

    # Initialize fitness score
    fitness_score = 0

    # Count the number of leading '1's until the first '0'
    for gene in decoded_individual:
        if gene == '1':
            fitness_score += 1
        else:
            break  # Stop counting at the first '0'

    # Calculate the penalty
    if len(decoded_individual) < MAX_LENGTH:
        penalty = 1.008 ** (MAX_LENGTH - len(decoded_individual))
    else:
        penalty = len(decoded_individual) - MAX_LENGTH

    # Return the final fitness score after applying the penalty
    return fitness_score - penalty


genes = ['0', '1']


def run_experiment(num_runs):
    results = []
    for run in range(num_runs):
        config = {
            'mutation_prob': random.uniform(0.01, 0.3),
            'delimited_mutation_prob': random.uniform(0.01, 0.3),
            'open_mutation_prob': random.uniform(0.01, 0.1),
            'capture_mutation_prob': random.uniform(0.001, 0.1),
            'delimiter_insert_prob': random.uniform(0.001, 0.2),
            'crossover_prob': random.uniform(0.30, 0.70),
            'base_gene_prob': random.uniform(0.30, 0.60),
            'elitism_ratio': 0.6,
            'max_individual_length': 40,
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
            'experiment_name': f'Bifurcation test {run}',
            'seed': GLOBAL_SEED,
            'capture_gene_prob': 0.15
        }

        # Initialize the GA
        ga = M_E_GA_Base(genes, leading_ones_fitness_function, **config)

        # Run the GA
        ga.run_algorithm()



        results.append({
            'run': run + 1,
            'config': config,
        })



    return results


# Example usage
num_runs = 300
experiment_results = run_experiment(num_runs)
