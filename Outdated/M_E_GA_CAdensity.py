import numpy as np
import matplotlib.pyplot as plt
import random
import cellpylib as cpl
from M_E_GA_Base_V2 import M_E_GA_Base




GLOBAL_SEED =            None
NUM_CYCLES =              1
MAX_GENERATIONS =         500
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

INITIAL_LENGTH = 100



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
            return True


def rule_from_bits(bits):
    """ Convert 8 bit number to decimal """
    return int("".join(str(x) for x in bits), 2)


def simulate_ca_with_rules(initial_config, rules, verbose=False):
    config = np.array(initial_config, dtype=int)
    result_matrix = [config]

    # Apply each rule exactly once
    for current_rule in rules:
        config = cpl.evolve(config.reshape(1, -1), timesteps=1,
                            apply_rule=lambda n, c, t: current_rule >> ((n[0] << 2) | (c << 1) | n[1]) & 1)[0]
        result_matrix.append(config)

    if verbose:
        plt.figure(figsize=(12, 6))
        plt.imshow(result_matrix, cmap='gray', interpolation='nearest')
        plt.title("CA Evolution Over Time Steps")
        plt.ylabel("Time Step")
        plt.xlabel("Cell Position")
        plt.show()

    return config



def evaluate_density_performance(config, final_config, penalties):
    # Majority class calculation
    majority_class = 0 if np.sum(config) <= len(config) / 2 else 1
    correct_cells = np.sum(final_config == majority_class)
    fitness_score = correct_cells / len(config) - penalties
    return max(0, fitness_score)


def create_ca_rules_from_genome(genome, rule_size=8):
    """Generate a list of rules from the genome."""
    rules = []
    for i in range(0, len(genome), rule_size):
        rule = genome[i:i + rule_size]
        if len(rule) < rule_size:
            rule += [0] * (rule_size - len(rule))  # Pad with zeros if not enough genes
        rules.append(int(''.join(map(str, rule)), 2))  # Convert to integer
    return rules


def apply_ca_rules(rules, initial_config, verbose=False):
    config = np.array(initial_config)
    history = [config.copy()]
    for rule in rules:
        config = simulate_ca_step(config, rule)
        history.append(config.copy())
    if verbose:
        plot_ca_history(history)
    return config


def simulate_ca_step(config, rule):
    """Apply one step of CA using the given rule on the current configuration."""
    next_config = np.zeros_like(config)
    for i in range(len(config)):
        neighborhood = (config[(i - 1) % len(config)] << 2) | (config[i] << 1) | config[(i + 1) % len(config)]
        next_config[i] = (rule >> neighborhood) & 1
    return next_config


def plot_ca_history(history):
    plt.figure(figsize=(10, len(history) // 2))
    plt.imshow(history, cmap='gray', interpolation='nearest')
    plt.title('Cellular Automaton Evolution')
    plt.xlabel('Cell Index')
    plt.ylabel('Time Step')
    plt.show()


def problem_specific_fitness_function(encoded_individual, ga_instance, verbose=False):
    decoded_individual = ga_instance.decode_organism(encoded_individual, format=True)
    decoded_individual = [int(gene) for gene in decoded_individual]

    rules = create_ca_rules_from_genome(decoded_individual)

    # Function to generate an initial configuration and evaluate it
    def evaluate_configuration():
        # Determine the dominant state and its probability
        dominant_state = random.choice([0, 1])
        dominant_prob = 0.52 if dominant_state == 1 else 0.48

        # Generate initial configuration with a biased random distribution
        initial_config = [1 if random.random() < dominant_prob else 0 for _ in range(INITIAL_LENGTH)]

        final_config = apply_ca_rules(rules, initial_config, verbose)

        # Determine the majority class in the initial configuration
        majority_class = int(sum(initial_config) > len(initial_config) / 2)

        # Manually count the number of cells in the final configuration that match the majority class
        correct_count = sum(1 for cell in final_config if cell == majority_class)

        # Calculate fitness score
        return correct_count / len(final_config)

    # Evaluate the fitness for two different initial configurations
    fitness_score_1 = evaluate_configuration()
    fitness_score_2 = evaluate_configuration()

    # Average the fitness scores from both evaluations
    averaged_fitness_score = (fitness_score_1 + fitness_score_2) / 2

    # Calculate penalties for incomplete rules
    penalty = (len(decoded_individual) % 8) * 1

    # Final fitness score after applying penalty
    final_fitness_score = max(0, averaged_fitness_score - penalty)

    update = update_best_organism(encoded_individual, final_fitness_score, verbose=True)
    if update and verbose:
        print("Updated best organism with fitness:", final_fitness_score)

    return final_fitness_score, {}


# Update GA configuration to use the new fitness function

# Gene set and configuration for the GA
GENES = ['0', '1']  # Binary gene set for initial conditions of CA


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
    genes=GENES,
    fitness_function=problem_specific_fitness_function,
    **config
)

# Execution and Result Analysis
ga.run_algorithm()
best_fitness = max(ga.fitness_scores)
best_index = ga.fitness_scores.index(best_fitness)
best_solution = ga.decode_organism(ga.population[best_index], format=True)
print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")
