import random
from M_E_GA_Base_V2 import M_E_GA_Base

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)  # Ensuring reproducibility

# Cellular Automaton Simulation Function
def simulate_ca(rule, initial_config):
    config = initial_config[:]
    for _ in range(100):  # Arbitrary number of steps
        new_config = []
        for i in range(len(config)):
            left = config[i - 1] if i > 0 else config[-1]
            center = config[i]
            right = config[i + 1] if i < len(config) - 1 else config[0]
            neighborhood = (left << 2) | (center << 1) | right
            new_config.append((rule >> neighborhood) & 1)
        config = new_config
    return config

# Cellular Automaton Fitness Function
def ca_fitness_function(organism, trials=100, config_length=100):
    rules = [int(''.join(map(str, organism[i:i+8])), 2) for i in range(0, len(organism), 8)]
    rule_fitnesses = []
    for rule in rules:
        correct_classifications = 0
        for _ in range(trials):
            initial_config = [random.randint(0, 1) for _ in range(config_length)]
            majority = sum(initial_config) > config_length / 2
            final_config = simulate_ca(rule, initial_config)
            predicted_majority = sum(final_config) > config_length / 2
            if majority == predicted_majority:
                correct_classifications += 1
        rule_fitnesses.append(correct_classifications / trials)
    return sum(rule_fitnesses) / len(rule_fitnesses)

# Define the Fitness Function for GA
def problem_specific_fitness_function(encoded_individual, encoding_manager):
    decoded_individual = encoding_manager.decode(encoded_individual)
    rules = []
    current_rule = []

    for gene in decoded_individual:
        if gene == 'Start':
            continue  # Skip and do nothing
        elif gene == 'End':
            continue  # Skip and do nothing
        else:
            current_rule.append(gene)
            if len(current_rule) == 8:  # Check if the rule is complete (8 bits)
                rules.extend(current_rule)
                current_rule = []  # Reset for next rule

    fitness_score = ca_fitness_function(rules)
    return fitness_score, {}

# Define Gene Set, Configuration, and Initialization
genes = [0, 1]  # Binary gene set for cellular automaton rules

config = {
    'mutation_prob': 0.05,
    'delimited_mutation_prob': 0.05,
    'open_mutation_prob': 0.001,
    'capture_mutation_prob': 0.0001,
    'delimiter_insert_prob': 0.00001,
    'crossover_prob': 0.7,
    'elitism_ratio': 0.05,
    'base_gene_prob': 0.98,
    'max_individual_length': 60,
    'population_size': 500,
    'num_parents': 90,
    'max_generations': 100,
    'delimiters': True,
    'delimiter_space': 2,
    'logging': True,
    'generation_logging': True,
    'mutation_logging': False,
    'crossover_logging': False,
    'individual_logging': False,
    'seed': GLOBAL_SEED
}

ga = M_E_GA_Base(
    genes=genes,
    fitness_function=problem_specific_fitness_function,
    **config
)

# Execution and Result Analysis
ga.run_algorithm()
best_fitness = max(ga.fitness_scores)
best_index = ga.fitness_scores.index(best_fitness)
best_solution = ga.decode_organism(ga.population[best_index], format=True)
print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")
