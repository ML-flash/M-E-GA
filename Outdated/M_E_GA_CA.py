import random
import math
import numpy as np
from M_E_GA_Base_V2 import M_E_GA_Base
import matplotlib.pyplot as plt
import cellpylib as cpl


GLOBAL_SEED =            None
NUM_CYCLES =              1
MAX_GENERATIONS =         5
random.seed(GLOBAL_SEED)


MUTATION_PROB =           0.02
DELIMITED_MUTATION_PROB = 0.01
OPEN_MUTATION_PROB =      0.007
CAPTURE_MUTATION_PROB =   0.001
DELIMITER_INSERT_PROB =   0.004
CROSSOVER_PROB =          .70
ELITISM_RATIO =           0.6
BASE_GENE_PROB =          0.45
MAX_INDIVIDUAL_LENGTH =   200
POPULATION_SIZE =         700
NUM_PARENTS =             150
DELIMITER_SPACE =         3
DELIMITERS =              False


LOGGING =                 False
GENERATION_LOGGING =      False
MUTATION_LOGGING =        True
CROSSOVER_LOGGING =       False
INDIVIDUAL_LOGGING =      False



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

# Helper functions for logic operations
def logic_and(inputs):
    return int(inputs[0] and inputs[1])

def logic_or(inputs):
    return int(inputs[0] or inputs[1])

def logic_not(inputs):
    return int(not inputs[0])


def simulate_ca_np(rule, initial_config, steps=20, verbose=False):
    config = np.array(initial_config, dtype=int)
    result_matrix = np.zeros((steps + 1, len(config)), dtype=int)
    result_matrix[0] = config

    for step in range(1, steps + 1):
        left = np.roll(config, 1)
        right = np.roll(config, -1)
        neighborhood = (left << 2) | (config << 1) | right
        config = (rule >> neighborhood) & 1
        result_matrix[step] = config

    if verbose:
        plt.figure(figsize=(10, 5))
        plt.imshow(result_matrix, cmap='gray_r', interpolation='nearest')  # Using 'gray_r' for reversed grayscale
        plt.title(f'Cellular Automaton Rule {rule} Evolution')
        plt.xlabel('Cell Index')
        plt.ylabel('Step')
        plt.colorbar(label='Cell State')
        plt.show()

    return result_matrix


def evaluate_logic_operation_np(config, rule, logic_funcs, output_indices, lane_width=3, steps=80, penalty_rate=0.01, verbose=False):
    total_correct = 0
    penalties = 0
    input_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    total_possible = len(input_combinations) * len(logic_funcs)

    for inputs in input_combinations:
        test_config = np.pad(config, (lane_width, lane_width), constant_values=(0, 0))  # pad to handle boundary conditions
        mid_index = len(test_config) // 2
        test_config[mid_index - 1: mid_index + 1] = inputs
        final_config = simulate_ca_np(rule, test_config, steps, verbose)

        for func, idx in zip(logic_funcs, output_indices):
            # Adjust index for padding
            idx += lane_width
            if idx >= len(final_config[-1]):
                continue  # skip if index is out of bounds
            expected = func(inputs)
            result = final_config[-1][idx]
            if result == expected:
                total_correct += 1

            # Check for noise in the lane
            start_idx = max(0, idx - lane_width)
            end_idx = min(len(final_config[-1]), idx + lane_width + 1)
            expected_state = 1 - result
            lane_noise = np.sum(final_config[-1][start_idx:end_idx] != expected_state) - (final_config[-1][idx] != expected_state)
            penalties += lane_noise
            if verbose and lane_noise > 0:
                print(f"Noise detected near output index {idx} for input ({inputs}).")

    correct_ratio = total_correct / total_possible
    fitness_score = correct_ratio - (penalties * penalty_rate)
    return max(0, fitness_score)

def problem_specific_fitness_function(encoded_individual, ga_instance, required_length=200, verbose=False):
    decoded_individual = ga_instance.decode_organism(encoded_individual, format=True)
    decoded_individual = np.array([int(gene) for gene in decoded_individual], dtype=int)

    # Calculate initial length of the individual
    original_length = len(decoded_individual)

    # Adjust the length of the individual
    if original_length < required_length:
        decoded_individual = np.pad(decoded_individual, (0, required_length - original_length), 'constant')
    elif original_length > required_length:
        # Optionally, you could trim the individual to required length if it exceeds, or let it be and penalize based on excess length
        # decoded_individual = decoded_individual[:required_length]
        pass

    # Calculate the penalty for exceeding the required length
    excess_length_penalty = max(0, original_length - required_length) * 0.005  # Example penalty rate per excess gene

    gate_positions = [required_length // 2, required_length // 2 - 7, required_length // 2 + 7]
    fitness_score = evaluate_logic_operation_np(decoded_individual, 110, [logic_and, logic_or, logic_not], gate_positions, verbose=verbose)

    # Subtract penalties from the fitness score
    fitness_score -= excess_length_penalty

    update_best_organism(encoded_individual, fitness_score, verbose=True)
    return max(0, fitness_score), {}


# Gene set and configuration for the GA
GENES = ['0', '1']  # Binary gene set for initial conditions of CA

# Define Gene Set, Configuration, and Initialization
genes = [0, 1]  # Binary gene set for cellular automaton rules

config = {
        'mutation_prob': MUTATION_PROB,
    'delimited_mutation_prob': DELIMITED_MUTATION_PROB,
    'open_mutation_prob': OPEN_MUTATION_PROB,
    'capture_mutation_prob': CAPTURE_MUTATION_PROB,
    'delimiter_insert_prob': DELIMITER_INSERT_PROB,
    'crossover_prob': CROSSOVER_PROB,
    'elitism_ratio': ELITISM_RATIO,
    'base_gene_prob': BASE_GENE_PROB,
    'max_individual_length': MAX_INDIVIDUAL_LENGTH,
    'population_size': POPULATION_SIZE,
    'num_parents': NUM_PARENTS,
    'max_generations': MAX_GENERATIONS,
    'delimiters': DELIMITERS,
    'delimiter_space': DELIMITER_SPACE,
    'logging': LOGGING,
    'generation_logging': GENERATION_LOGGING,
    'mutation_logging': MUTATION_LOGGING,
    'crossover_logging': CROSSOVER_LOGGING,
    'individual_logging': INDIVIDUAL_LOGGING,
    'seed': GLOBAL_SEED
}

ga = M_E_GA_Base(
    genes=genes,
    fitness_function=problem_specific_fitness_function,
    **config
)

# Execution and Result Analysis
ga.run_algorithm()

final_fitness, _ = problem_specific_fitness_function(best_organism['genome'], ga, required_length=200, verbose=True)
