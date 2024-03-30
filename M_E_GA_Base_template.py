import random
from M_E_GA_Base_V2 import M_E_GA_Base

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED) # Ensure seed is used and passed properly as this is research code and reproducability is critical.

# Step 1: Define the Fitness Function
def problem_specific_fitness_function(encoded_individual,encoding_manager):
    """
    Note: Mutabel encodign uses delimiters in the encodings and you need to adapt the
    fitness function to account for them see the pattern below.
    Calculates the fitness score for an individual, considering delimited regions.
    
    Parameters:
    - encoded_individual: The encoded gene sequence of the individual. used to calculate fitness
    -encoding_manager: used to decode individual decoded_individual = encoding_manager.decode(encoded_individual)
    
    
    
    Returns:
    - A numeric score representing the individual's fitness.
    """
    
    #Pay attention the below handling of delimiters if you dont things will break they are there for a reason
    #Delimiters are default genes you do not need to add them you just need to build the fitness function to be aware they exist
    # Initialize fitness score and flags
    fitness_score = 0
    in_delimited_region = False  # Flag to track if we're within a delimited region
    
    for gene in decoded_individual:
        if gene == 'Start':
            in_delimited_region = True  # Enter delimited region
        elif gene == 'End':
            in_delimited_region = False  # Exit delimited region
        else:
            # Process gene value based on whether it's within a delimited region
            gene_value = ...  # Determine the value of the gene based on your problem
            if in_delimited_region:
                # Optionally modify gene value if it's within a delimited region
                gene_value += ...  # Additional value or modification for genes within delimited regions
                
            fitness_score += gene_value  # Add gene value to the total fitness score
    
    # Apply any penalties or bonuses based on the problem's constraints
    ...
    
    return fitness_score


# Step 2: Define the Gene Set
genes = ['gene1', 'gene2', 'gene3', ...]  # Define genes relevant to the problem

# Step 3: Configuration
# all of the below settings are critical for the GA to function properly please dont exclude them.
#delimiters are always True they are part of mutabel encoding and dont apply to any speciffic GA but are general in nature
config = {
    'mutation_prob': 0.05,  # Probability of a standard mutation
    'delimited_mutation_prob': 0.05,  # Mutation probability within delimited segments
    'open_mutation_prob': 0.001,  # Probability of expanding a compressed gene segment
    'capture_mutation_prob': 0.0001,  # Probability of capturing a segment into a single codon
    'delimiter_insert_prob': 0.00001,  # Probability of inserting a new pair of start/end delimiters around a gene
    'crossover_prob': 0.7,  # Probability of crossover between two individuals
    'elitism_ratio': 0.05,  # Proportion of top individuals that pass directly to the next generation
    'base_gene_prob': 0.98,  # Probability of choosing a base gene over a captured codon during mutation
    'max_individual_length': 60,  # Maximum number of genes in an individual
    'population_size': 500,  # Number of individuals in the population
    'num_parents': 90,  # Number of parents selected for breeding
    'max_generations': 100,  # Maximum number of generations to evolve
    'delimiters': True,  # Use delimiters to denote gene segments
    'delimiter_space': 2,  # Minimum number of genes between delimiters when inserted
    'logging': True,  # Enable logging
    'generation_logging': True,  # Log information about each generation
    'mutation_logging': False,  # Disable logging of mutation information
    'crossover_logging': False,  # Disable logging of crossover information
    'individual_logging': False,  # Disable logging of individual information
    'seed': GLOBAL_SEED          #seed for reproducability
}
# Step 4: GA Initialization
ga = M_E_GA_Base(
    genes=genes,
    fitness_function=problem_specific_fitness_function,
    **config  # Unpack all configuration parameters
)

# Step 5: Execution
ga.run_algorithm()

# Step 6: Result Analysis
best_fitness = max(ga.fitness_scores)
best_index = ga.fitness_scores.index(best_fitness)
best_solution = ga.decode_organism(ga.population[best_index], format=True)
print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")



'''Full Instructions for Using the GA Template with Mutable Encoding
Step 1: Define the Fitness Function
The fitness function is crucial as it evaluates the suitability of each individual within the population in solving the given problem. Implement logic that accurately reflects your problem's objectives and constraints.

python
Copy code
def problem_specific_fitness_function(individual):
    """
    Calculates the fitness score for an individual based on the problem's objectives and constraints.
    
    Parameters:
    - individual (list): Represents the gene sequence of the individual.
    
    Returns:
    - float: The fitness score of the individual.
    """
    fitness_score = 0
    for gene_id in individual:
        gene_value = ...  # Logic to determine the value of each gene
        fitness_score += gene_value
    return fitness_score
Step 2: Define the Gene Set
Identify and list all the genes relevant to your problem, ensuring each gene has a unique identifier. This gene set will form the basis for constructing individuals in the population.

python
Copy code
genes = ['gene1', 'gene2', 'gene3', ...]
Step 3: Configuration
Set up the GA's parameters to define its behavior. It's essential to include all the listed parameters to fully utilize the mutable encoding features.

python
Copy code
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
    'individual_logging': False
    }
Step 4: GA Initialization
Initialize the GA by creating an instance of M_E_GA_Base, providing it with the gene set, fitness function, and configuration parameters.

python
Copy code
ga = M_E_GA_Base(
    genes=genes,
    fitness_function=problem_specific_fitness_function,
    **config  # Unpacks all configuration parameters
)
Step 5: Execution
Run the GA to evolve solutions. The run_algorithm() method iterates through generations, seeking to improve the overall fitness of the population.

python
Copy code
ga.run_algorithm()
Step 6: Result Analysis
After the GA execution completes, identify and analyze the best solution(s) from the final population. Use the decode_organism method, if necessary, to interpret the gene sequence of the best individual in a readable format.

python
Copy code
best_fitness = max(ga.fitness_scores)
best_index = ga.fitness_scores.index(best_fitness)
best_solution = ga.decode_organism(ga.population[best_index], format=True)
print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")
Additional Considerations
Fitness Function: Tailor this function to your specific problem, ensuring it accurately assesses each individual's fitness.
Decoding: Understand the decoding process, as it's essential for the fitness function to evaluate the decoded gene sequence of each individual.
Configuration Parameters: Experiment with different settings to find the optimal configuration for your problem. Mutable encoding features like segment capture and expansion can significantly impact the GA's performance.
Continuous Improvement: After each GA run, analyze the results to gain insights into the algorithm's performance. Use this information to refine the fitness function, gene set, and GA parameters for improved results in subsequent runs.
By following these instructions and considering the additional tips, you can effectively adapt the GA template with mutable encoding to solve various optimization problems, taking advantage of the dynamic and flexible nature of mutable encoding.
'''



