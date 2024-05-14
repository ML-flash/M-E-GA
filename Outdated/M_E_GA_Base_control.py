from M_E_GA_Base import M_E_GA_Base

# Define the fitness function for the knapsack problem
def knapsack_fitness_function(individual, gene_info, max_weight):
    total_weight = total_value = 0
    for gene_id in individual:
        item = gene_info.get(gene_id)
        if item:
            total_weight += item['weight']
            total_value += item['value']
            if total_weight > max_weight:
                return 0  # Penalize over-weight solutions
    return total_value

# Original genes (items) for the knapsack problem
original_genes = [
    {'id': 'item1', 'weight': 50, 'value': 60},
    {'id': 'item2', 'weight': 20, 'value': 100},
    {'id': 'item3', 'weight': 30, 'value': 120},
]

# Preprocess to extract gene IDs and create a dictionary for gene info
gene_ids = [gene['id'] for gene in original_genes]
gene_info = {gene['id']: gene for gene in original_genes}

# Define the experiment configuration
config = {
    #required Settings for the GA to work
    'mutation_prob': 0.05,
    'delimited_mutation_prob': 0.05,
    'open_mutation_prob': 0.001,
    'capture_mutation_prob': 0.0001,
    'delimiter_insert_prob': 0.00001,
    'crossover_prob': 0.70,
    'elitism_ratio': 0.05,
    'base_gene_prob': 0.98,
    'max_individual_length': 60,
    'population_size': 500,
    'num_parents': 90,
    'max_generations': 100,
    'delimiters': True,
    'delimiter_space': 2,
    # Specific to the knapsack problem
    'max_weight': 279,  
}

# Initialize the genetic algorithm with the provided configuration and fitness function
ga = M_E_GA_Base(
    genes=gene_ids,  # Use only gene IDs
    fitness_function=lambda individual: knapsack_fitness_function(individual, gene_info, config['max_weight']),
    mutation_prob=config['mutation_prob'],
    delimited_mutation_prob=config['delimited_mutation_prob'],
    open_mutation_prob=config['open_mutation_prob'],
    capture_mutation_prob=config['capture_mutation_prob'],
    delimiter_insert_prob=config['delimiter_insert_prob'],
    crossover_prob=config['crossover_prob'],
    elitism_ratio=config['elitism_ratio'],
    base_gene_prob=config['base_gene_prob'],
    max_individual_length=config['max_individual_length'],
    population_size=config['population_size'],
    num_parents=config['num_parents'],
    max_generations=config['max_generations'],
    delimiters=config['delimiters'],
    delimiter_space=config['delimiter_space'],
    logging=True,
    experiment_name="KnapsackExperiment"
)

# Run the genetic algorithm experiment
ga.run_algorithm()

# Display the best solution from the final population
best_fitness = max(ga.fitness_scores)
best_index = ga.fitness_scores.index(best_fitness)
best_solution = ga.decode_organism(ga.population[best_index], format=True)
print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")
