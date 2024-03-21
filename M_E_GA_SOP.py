# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:12:59 2024

@author: Matt Andrews
"""

import gzip
import random
from M_E_GA_Base_V2 import M_E_GA_Base

# Global seed for reproducibility
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)



def load_sop_data(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        lines = file.readlines()

    distance_matrix = []
    dimension = None
    edge_weight_section_started = False
    precedence_constraints = []  # Initialize the list to store precedence constraints

    for line in lines:
        line = line.strip()

        if line.startswith('EOF'):
            break
        elif line.startswith("DIMENSION:"):
            dimension = int(line.split(":")[1])
            print("Dimension specified in the file:", dimension)
        elif line.startswith("EDGE_WEIGHT_SECTION") and dimension is not None:
            edge_weight_section_started = True
        elif edge_weight_section_started:
            if line.isdigit() and int(line) == dimension:
                continue  # Skipping the dimension line if present
            row = [int(x) if x != "1000000" else float('inf') for x in line.split()]
            distance_matrix.append(row)

    # Deriving precedence constraints from infeasible paths
    for i, row in enumerate(distance_matrix):
        for j, distance in enumerate(row):
            if distance == float('inf'):
                precedence_constraints.append((i, j))

    if len(distance_matrix) != dimension:
        raise ValueError("Dimension specified does not match the number of rows read.")
    else:
        print("Dimension check passed successfully.")

    print("Matrix read successfully. First row:", distance_matrix[0])
    print("Precedence constraints derived:", precedence_constraints)

    genes = list(range(dimension))
    print(precedence_constraints)

    return genes, distance_matrix, precedence_constraints



def sop_fitness_function(encoded_individual, encoding_manager, distance_matrix, precedence_constraints):
    organism_length = len(encoded_individual)
    decoded_individual = encoding_manager.decode(encoded_individual)
    total_distance = 0
    precedence_violations = 0
    duplicate_penalty = 0

    # Encourage compressed encoding
    size_penalty = organism_length * 2  # Adjust the factor as needed

    max_allowed_length = len(distance_matrix)
    filtered_individual = [gene for gene in decoded_individual if gene not in ['Start', 'End']]
    
    if len(filtered_individual)  >max_allowed_length:
        size_penalty += (len(filtered_individual) - max_allowed_length) * 50
    if len(filtered_individual) > max_allowed_length:
        size_penalty += (max_allowed_length - len(filtered_individual)) * 50
    
    visited_nodes = set()
    missing_nodes_penalty = 0

    for i, gene in enumerate(filtered_individual):
        # Duplicate check
        if gene in visited_nodes:
            duplicate_penalty += 10000.1009
        else:
            visited_nodes.add(gene)

        # Distance calculation and infeasibility check
        if i < len(filtered_individual) - 1:
            next_gene = filtered_individual[i + 1]
            path_distance = distance_matrix[gene][next_gene]
            total_distance += path_distance if path_distance < float('inf') else 10000.1009

    # Ensure all nodes are visited at least once
    missing_nodes = set(range(max_allowed_length)) - visited_nodes
    missing_nodes_penalty += len(missing_nodes) * 10000.1009

    # Precedence constraint violations
    for a, b in precedence_constraints:
        if a in visited_nodes and b in visited_nodes:
            if filtered_individual.index(a) > filtered_individual.index(b):
                precedence_violations += 10000.1009

    penalties = total_distance + size_penalty + duplicate_penalty + missing_nodes_penalty + precedence_violations
    return -penalties  # Negative for lower (better) scores




def evaluate_sop_solution(solution, distance_matrix, precedence_constraints, verbose=True):
    total_distance = 0
    violations = []
    visited_nodes = set()

    if verbose:
        print(f"Evaluating solution: {solution}")

    for i in range(len(solution)):
        from_node = solution[i]
        to_node = solution[i + 1] if i + 1 < len(solution) else None

        # Mark the current node as visited
        visited_nodes.add(from_node)

        # Check for path feasibility to the next node if it exists
        if to_node is not None:
            if verbose:
                print(f"Checking path from {from_node} to {to_node}")

            if 0 <= from_node < len(distance_matrix) and 0 <= to_node < len(distance_matrix):
                path_distance = distance_matrix[from_node][to_node]
                if path_distance < float('inf'):  # Path is feasible
                    total_distance += path_distance
                else:
                    violation_message = f"Infeasible path from {from_node} to {to_node}"
                    violations.append(violation_message)
                    if verbose:
                        print(violation_message)
            else:
                violation_message = f"Index out of bounds: from_node={from_node}, to_node={to_node}"
                violations.append(violation_message)
                if verbose:
                    print(violation_message)

    # Check for nodes not visited
    all_nodes = set(range(len(distance_matrix)))
    not_visited = all_nodes - visited_nodes
    for node in not_visited:
        violation_message = f"Node {node} not visited."
        violations.append(violation_message)
        if verbose:
            print(violation_message)

    # Check precedence constraints
    for constraint in precedence_constraints:
        if verbose:
            print(f"Checking precedence constraint: {constraint}")
        if constraint[0] in visited_nodes and constraint[1] in visited_nodes:
            if solution.index(constraint[0]) > solution.index(constraint[1]):
                violation_message = f"Precedence violation: Node {constraint[0]} must precede Node {constraint[1]}"
                violations.append(violation_message)
                if verbose:
                    print(violation_message)
        elif verbose:
            print(f"Precedence constraint {constraint} not applicable - one or both nodes not in solution.")

    return total_distance, violations





# Load SOP data
sop_file_path = r"C:\Users\Matt\AppData\Local\Temp\0d5e55ec-dd10-4f62-8a99-c2e92744c462_ALL_sop.tar.462\br17.12.sop.gz"
genes, distance_matrix, precedence_constraints = load_sop_data(sop_file_path)

# Wrapper function for sop_fitness_function
def sop_fitness_wrapper(encoded_individual, encoding_manager):
    # Use distance_matrix and precedence_constraints from the outer scope
    return sop_fitness_function(encoded_individual, encoding_manager, distance_matrix, precedence_constraints)

# GA Configuration
config = {
    'mutation_prob': 0.015,  # Probability of a standard mutation
    'delimited_mutation_prob': 0.012,  # Mutation probability within delimited segments
    'open_mutation_prob':      0.007,  # Probability of expanding a compressed gene segment
    'capture_mutation_prob':   0.001,  # Probability of capturing a segment into a single codon
    'delimiter_insert_prob':   0.004,  # Probability of inserting a new pair of start/end delimiters around a gene
    'crossover_prob':          1.00,  # Probability of crossover between two individuals
    'elitism_ratio':           0.05,  # Proportion of top individuals that pass directly to the next generation
    'base_gene_prob':          0.60,  # Probability of choosing a base gene over a captured codon during mutation
    'max_individual_length':   60,  # Maximum number of genes in an individual
    'population_size':         500,  # Number of individuals in the population
    'num_parents':             100,  # Number of parents selected for breeding
    'max_generations':         500,  # Maximum number of generations to evolve
    'delimiters': False,  # Use delimiters to denote gene segments
    'delimiter_space': 5,  # Minimum number of genes between delimiters when inserted
    'logging': False,  # Enable logging
    'generation_logging': True,  # Log information about each generation
    'mutation_logging': False,  # Disable logging of mutation information
    'crossover_logging': False,  # Disable logging of crossover information
    'individual_logging': False,  # Disable logging of individual information
    'seed': GLOBAL_SEED          #seed for reproducability
}

# GA Initialization with the wrapper function as the fitness function
ga = M_E_GA_Base(
    genes=genes,
    fitness_function=sop_fitness_wrapper,  # Pass the wrapper function
    **config
)

# GA Execution
ga.run_algorithm()

best_fitness = max(ga.fitness_scores)
best_index = ga.fitness_scores.index(best_fitness)
best_solution_encoded = ga.population[best_index]  # This is the encoded best solution

# Decode the best solution (assuming your GA uses an encoding mechanism)
best_solution_decoded = ga.decode_organism(best_solution_encoded, format= True)
print(best_solution_decoded)

# Evaluate the filtered solution against SOP data
total_distance, violations = evaluate_sop_solution(best_solution_decoded, distance_matrix, precedence_constraints)

# Present the Evaluation Results
print(f"Best Solution (Filtered): {best_solution_decoded}")
print(f"Total Distance of Best Solution: {total_distance}")
if violations:
    print(f"Number of Precedence Constraint Violations: {len(violations)}")
    for violation in violations:
        print(f"Violation: Node {violation[0]} must precede Node {violation[1]}")
else:
    print("No Precedence Constraint Violations")
    
print(ga.encoding_manager.encodings)