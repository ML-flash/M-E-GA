# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:38:05 2024

@author: Matt Andrews
"""

import random
from M_E_GA_Base_V2 import M_E_GA_Base



GLOBAL_SEED = 3010
random.seed(GLOBAL_SEED)
num_cycles=          2
MAX_GENERATIONS =    500

MUTATION_PROB =           0.015   
DELIMITED_MUTATION_PROB = 0.014
OPEN_MUTATION_PROB =      0.007
CAPTURE_MUTATION_PROB =   0.001 
DELIMITER_INSERT_PROB =   0.004
CROSSOVER_PROB =          0.90
ELITISM_RATIO =           0.05
BASE_GENE_PROB =          0.5
MAX_INDIVIDUAL_LENGTH =   60
POPULATION_SIZE =         600
NUM_PARENTS =             100
DELIMITER_SPACE =         3
DELIMITERS =              False

LOGGING =                 True
GENERATION_LOGGING =      True
MUTATION_LOGGING =        False
CROSSOVER_LOGGING =       False
INDIVIDUAL_LOGGING =      True

HARD_LIMIT = 200  # Hard limit for the organism size.
BASE_PENALTY = 2000  # Base for the exponential penalty, adjust as needed
MIN_PENALTY = 700  #Lowest the penalty goes at max efficency


GENES = ['R', 'L', 'U', 'D', 'F', 'B', 'H', 'P']
directions = {'R': (1, 0, 0), 'L': (-1, 0, 0), 'U': (0, 1, 0), 'D': (0, -1, 0), 'F': (0, 0, 1), 'B': (0, 0, -1)}

    
def calculate_penalty(encoded_length, decoded_length):
    if encoded_length == 0:  # Prevent division by zero
        return BASE_PENALTY

    efficiency_ratio = decoded_length / encoded_length
    penalty = BASE_PENALTY

    if decoded_length <= HARD_LIMIT:
        # As efficiency improves, reduce the penalty, but not below the min_penalty
        if efficiency_ratio > 1:
            penalty /= efficiency_ratio  # Decrease penalty for higher efficiency
            penalty = max(penalty, MIN_PENALTY)  # Enforce a minimum penalty
    else:
        # For decoded lengths exceeding the hard limit, increase penalty smoothly
        excess_ratio = decoded_length / HARD_LIMIT
        penalty *= excess_ratio  # Increase penalty based on how much the length exceeds the limit

    return penalty



def hp_model_fitness_function_3d(encoded_individual, encoding_manager):
    decoded_individual = encoding_manager.decode(encoded_individual)
    fitness_score = 0
    x, y, z = 0, 0, 0  # Initial 3D coordinates
    positions = {(x, y, z)}  # Track occupied 3D positions
    last_direction = None

    for gene in decoded_individual:
        if gene == 'Start':
            # Reset coordinates at the start of a new delimited region
            x, y, z = 0, 0, 0
        elif gene == 'End':
            continue
        elif gene in directions:
            dx, dy, dz = directions[gene]
            x += dx
            y += dy
            z += dz
            if (x, y, z) in positions and last_direction != (-dx, -dy, -dz):  # Check for overlaps, excluding immediate backtrack
                fitness_score -= 10  # Penalize overlaps
            positions.add((x, y, z))
            last_direction = (dx, dy, dz)
        elif gene == 'H':
            # Check for adjacent H-H contacts in 3D, excluding the last move's reverse direction
            adjacent_positions = [(x + dx, y + dy, z + dz) for dx, dy, dz in directions.values() if (dx, dy, dz) != last_direction]
            for pos in adjacent_positions:
                if pos in positions:
                    fitness_score += 1  # H-H contact found

    # Apply size penalty
    size_penalty = calculate_penalty(len(encoded_individual), len(decoded_individual))
    fitness_score -= (size_penalty + (len(encoded_individual) / 10))

    return (fitness_score + BASE_PENALTY)






class ExperimentGA(M_E_GA_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization if needed

    def instructor_phase(self):
        self.run_algorithm()
        return self.encoding_manager.encodings

    def student_phase(self, instructor_encodings):
        self.encoding_manager.integrate_uploaded_encodings(instructor_encodings, GENES)
        self.run_algorithm()

    def control_phase(self):
        self.run_algorithm()

def run_experiment(experiment_name, num_cycles, genes, fitness_function):
    for cycle in range(1, num_cycles + 1):
        print(f"\n--- Starting Experiment Cycle: {cycle} ---")

        instructor_ga = ExperimentGA(
           genes=genes,
           fitness_function=fitness_function,
           mutation_prob=           MUTATION_PROB,
           delimited_mutation_prob= DELIMITED_MUTATION_PROB,
           open_mutation_prob=      OPEN_MUTATION_PROB,
           capture_mutation_prob=   CAPTURE_MUTATION_PROB,
           delimiter_insert_prob=   DELIMITER_INSERT_PROB,
           crossover_prob=          CROSSOVER_PROB,
           elitism_ratio=           ELITISM_RATIO,
           base_gene_prob=          BASE_GENE_PROB,
           max_individual_length=   MAX_INDIVIDUAL_LENGTH,
           population_size=         POPULATION_SIZE,
           num_parents=             NUM_PARENTS,
           max_generations =        MAX_GENERATIONS,
           delimiters=              DELIMITERS,
           delimiter_space=        DELIMITER_SPACE,
           logging=                 LOGGING,
           generation_logging =     GENERATION_LOGGING, 
           mutation_logging =       MUTATION_LOGGING,
           crossover_logging =      CROSSOVER_LOGGING, 
           individual_logging =     INDIVIDUAL_LOGGING,
           experiment_name =        f"{experiment_name}_Instructor_Cycle_{cycle}",
           seed =                   GLOBAL_SEED
           )
        instructor_encodings = instructor_ga.instructor_phase()

        student_ga = ExperimentGA(
            genes=genes,
            fitness_function=        fitness_function,
            mutation_prob=           MUTATION_PROB,
            delimited_mutation_prob= DELIMITED_MUTATION_PROB,
            open_mutation_prob=      0.000,
            capture_mutation_prob=   0,
            delimiter_insert_prob=   0,
            crossover_prob=          .90,
            elitism_ratio=           ELITISM_RATIO,
            base_gene_prob=          BASE_GENE_PROB,
            max_individual_length=   MAX_INDIVIDUAL_LENGTH,
            population_size=         POPULATION_SIZE,
            num_parents=             NUM_PARENTS,
            max_generations =        MAX_GENERATIONS,
            delimiters=              False,
            delimiter_space=         0,
            logging =                LOGGING,
            generation_logging =     GENERATION_LOGGING, 
            mutation_logging =       MUTATION_LOGGING,
            crossover_logging =      CROSSOVER_LOGGING, 
            individual_logging =     INDIVIDUAL_LOGGING,
            experiment_name=         f"{experiment_name}_Student_Cycle_{cycle}",
            seed =                   GLOBAL_SEED
            )
        
        student_ga.student_phase(instructor_encodings)

        control_ga = ExperimentGA(
            genes=                   genes,
            fitness_function=        fitness_function,
            mutation_prob=           MUTATION_PROB,
            delimited_mutation_prob= DELIMITED_MUTATION_PROB,
            open_mutation_prob=      0,
            capture_mutation_prob=   0,
            delimiter_insert_prob=   0,
            crossover_prob=          CROSSOVER_PROB,
            elitism_ratio=           ELITISM_RATIO,
            base_gene_prob=          BASE_GENE_PROB,
            max_individual_length=   MAX_INDIVIDUAL_LENGTH,
            population_size=         POPULATION_SIZE,
            num_parents=             NUM_PARENTS,
            max_generations =        MAX_GENERATIONS,
            delimiters=              False,
            delimiter_space=         0,
            logging =                LOGGING,
            generation_logging =     GENERATION_LOGGING, 
            mutation_logging =       MUTATION_LOGGING,
            crossover_logging =      CROSSOVER_LOGGING, 
            individual_logging =     INDIVIDUAL_LOGGING,
            experiment_name=         f"{experiment_name}_Control_Cycle_{cycle}",
            seed =                   GLOBAL_SEED
            )
        control_ga.control_phase()

        # Log results and compare the phases
        print("\n--- Results Summary ---")
        compare_results(instructor_ga, student_ga, control_ga, cycle)

def compare_results(instructor_ga, student_ga, control_ga, cycle):
    # Implement your logic to compare and log results from each phase for the current cycle
    # Placeholder for demonstration purposes
    print(f"Results for Cycle {cycle}:\nInstructor Best Fitness: {max(instructor_ga.fitness_scores)}\nStudent Best Fitness: {max(student_ga.fitness_scores)}\nControl Best Fitness: {max(control_ga.fitness_scores)}")


if __name__ == '__main__':
    genes = GENES
    experiment_name = input("Your_Experiment_Name: ")
    run_experiment(experiment_name, num_cycles, genes, hp_model_fitness_function_3d)
