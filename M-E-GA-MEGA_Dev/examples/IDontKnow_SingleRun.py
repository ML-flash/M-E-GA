import random
from M_E_GA import M_E_GA_Base
from I_Dont_Know_Drop import IDontKnow

# Global configuration
VOLUME = 60
NUM_ITEMS = 1000
NUM_GROUPS = 20
MAX_SIZE = 500
MAX_WEIGHT = 75
MAX_DENSITY = 300
GLOBAL_SEED = None

random.seed(GLOBAL_SEED)

best_organism = {
    "genome": None,
    "fitness": float('-inf')
}

def update_best_organism(current_genome, current_fitness, verbose=True):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")

class PopulationFitnessEvaluator:
    """
    Population-level fitness evaluator for scenarios with temporal dependencies.
    
    CRITICAL: This evaluator maintains a single fitness function instance
    throughout the entire GA run. The fitness function state is NEVER
    reinitialized between individuals or generations to preserve temporal
    dependencies and accumulated state changes.
    """
    
    def __init__(self, fitness_function):
        self.fitness_function = fitness_function
        self.evaluation_count = 0  # Track total evaluations for debugging
        
        # Store the fitness function ID to verify it's never replaced
        self.fitness_function_id = id(fitness_function)
        
        print(f"PopulationFitnessEvaluator initialized with fitness function ID: {self.fitness_function_id}")
    
    def evaluate_individual(self, individual, ga_instance):
        """
        Evaluate a single individual using the persistent fitness function.
        
        The fitness function state carries forward from all previous evaluations.
        """
        # Verify we're still using the same fitness function instance
        current_id = id(self.fitness_function)
        if current_id != self.fitness_function_id:
            raise RuntimeError(f"Fitness function instance changed! Original: {self.fitness_function_id}, Current: {current_id}")
        
        self.evaluation_count += 1
        return self.fitness_function.compute(individual, ga_instance)
    
    def evaluate(self, population, ga_instance):
        """
        Evaluate the entire population sequentially using the persistent fitness function.
        
        Each evaluation modifies the fitness landscape, so:
        1. Order matters (sequential evaluation required)
        2. Threading is impossible (would break temporal dependencies)
        3. State persists across all evaluations in all generations
        
        :param population: List of encoded organisms
        :param ga_instance: The GA instance for context
        :return: List of fitness scores
        """
        generation = getattr(ga_instance, 'current_generation', 'Unknown')
        print(f"Total evaluations so far: {self.evaluation_count}")
        
        fitness_scores = []
        for individual in population:
            fitness = self.evaluate_individual(individual, ga_instance)
            fitness_scores.append(fitness)
        
        return fitness_scores

# CRITICAL: Initialize the fitness function ONCE and reuse throughout entire GA run
# This single instance maintains all temporal dependencies and accumulated state
# The fitness function state is NEVER reset between individuals or generations
fitness_function = IDontKnow(
    volume=VOLUME,
    num_items=NUM_ITEMS,
    num_groups=NUM_GROUPS,
    update_best_func=update_best_organism,
    max_size=MAX_SIZE,
    max_weight=MAX_WEIGHT,
    max_density=MAX_DENSITY
)
genes = fitness_function.genes

print(f"Fitness function initialized with ID: {id(fitness_function)}")
print(f"Fitness function will persist throughout entire GA run maintaining temporal state")

# Create the population fitness evaluator
population_evaluator = PopulationFitnessEvaluator(fitness_function)

config = {
    'mutation_prob': 0.15,
    'delimited_mutation_prob': 0.10,
    'open_mutation_prob': 0.10,
    'metagene_mutation_prob': 0.06,  
    'delimiter_insert_prob': 0.06,
    'delimit_delete_prob': 0.06,
    'crossover_prob': 0.0,
    'elitism_ratio': 0.07,
    'base_gene_prob': 0.45,
    'metagene_prob': 0.008,
    'max_individual_length': 40,
    'population_size': 500,
    'num_parents': 300,
    'max_generations': 8000,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': True,
    'generation_logging': True,
    'mutation_logging': True,
    'crossover_logging': True,
    'individual_logging': True,
    'seed': GLOBAL_SEED,
    'lru_cache_size': 80
}

# Initialize the GA with the population evaluator instead of individual fitness function
ga = M_E_GA_Base(
    genes=genes,
    fitness_function=None,  # Not used when fitness_evaluator is provided
    fitness_evaluator=population_evaluator,  # Use population-level evaluator
    **config
)

# Run the GA
print("Starting GA run with persistent fitness function...")
ga.run_algorithm()

# Verify fitness function persistence
print(f"\nGA completed. Fitness function ID verification:")
print(f"  Original fitness function ID: {id(fitness_function)}")
print(f"  Evaluator's fitness function ID: {id(population_evaluator.fitness_function)}")
print(f"  Total evaluations performed: {population_evaluator.evaluation_count}")
print(f"  Fitness function state preserved: {id(fitness_function) == id(population_evaluator.fitness_function)}")

# Find the best solution
best_genome = best_organism["genome"]
best_fitness = best_organism["fitness"]
best_solution_decoded = ga.decode_organism(best_genome, format=True)

print(f'\nResults:')
print(f'Length of best solution: {len(best_solution_decoded)}')
print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
print(f'Length of best genome: {len(best_organism["genome"])}')
print(f"Best Genome (Encoded): {best_genome}")
