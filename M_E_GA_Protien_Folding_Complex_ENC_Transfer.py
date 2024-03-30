# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 22:56:55 2024

@author: Matt Andrews
"""

import random
from M_E_GA_Base_V2 import M_E_GA_Base
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


GLOBAL_SEED = 3010
random.seed(GLOBAL_SEED)
num_cycles=          1
MAX_GENERATIONS =    500
MAX_LENGTH = 250

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

HARD_LIMIT = 200  # Hard limit for the organism size.
BASE_PENALTY = 2000  # Base for the exponential penalty, adjust as needed
MIN_PENALTY = 1950  #Lowest the penalty goes at max efficency


LOGGING =                 True
GENERATION_LOGGING =      True
MUTATION_LOGGING =        False
CROSSOVER_LOGGING =       False
INDIVIDUAL_LOGGING =      True



GENES = ['R', 'L', 'U', 'D', 'F', 'B','H','P']
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

# Fitness function to score the HP model of protein folding
def protein_folding_fitness_function(encoded_individual, encoding_manager):
    encoded_length = len(encoded_individual)
    decoded_individual = encoding_manager.decode(encoded_individual)
    decoded_length = len(decoded_individual)
    fitness_score = 0
    x, y, z = 0, 0, 0
    volume_bound = 2  # Define the volume boundary
    visited_positions = {(x, y, z): None}  # Track positions with gene type
    hh_contacts = 0  # Track hydrophobic contacts
    penalty = 0  # Penalty for invalid actions
    expecting_move = False  # Toggle between expecting a move or a gene
    prev_gene = None  # Initialize prev_gene

    for gene in decoded_individual:
        if gene == 'Start':
            encoded_length -= 1
            decoded_length -= 1
        elif gene == 'End':
            encoded_length -= 1
            decoded_length -= 1
        if expecting_move:
            if gene in directions:  # Valid move after a gene
                dx, dy, dz = directions[gene]
                new_pos = (x + dx, y + dy, z + dz)

                # Check for volume boundaries and overlaps
                if not (-volume_bound <= new_pos[0] <= volume_bound and
                        -volume_bound <= new_pos[1] <= volume_bound and
                        -volume_bound <= new_pos[2] <= volume_bound) or new_pos in visited_positions:
                    fitness_score -= penalty  # Apply penalty
                else:
                    # Place the prev_gene at the new position
                    if prev_gene:  # Ensure prev_gene is not None
                        visited_positions[new_pos] = prev_gene
                    x, y, z = new_pos  # Update the position

                expecting_move = False  # Next, expect a gene
            else:
                fitness_score -= penalty  # Penalty for consecutive genes
        else:
            if gene in ['H', 'P']:  # Expecting a gene, valid gene found
                prev_gene = gene  # Store this gene to place after a valid move
                expecting_move = True  # Next, expect a move
            else:
                fitness_score -= penalty  # Penalty for consecutive moves or invalid gene

    # Calculate fitness based on H-H contacts and apply penalties
    for pos, gene_type in visited_positions.items():
        if gene_type == 'H':
            # Check adjacent positions for hydrophobic contacts
            for d in directions.values():
                adjacent_pos = (pos[0] + d[0], pos[1] + d[1], pos[2] + d[2])
                if adjacent_pos in visited_positions and visited_positions[adjacent_pos] == 'H':
                    hh_contacts += 1

    fitness_score += hh_contacts / 2  # Each H-H contact is counted twice, so divide by 2
    fitness_score -= calculate_penalty(encoded_length, decoded_length)
    return fitness_score + BASE_PENALTY







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
           after_population_selection=capture_best_organism,
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
            fitness_function=fitness_function,
            after_population_selection=capture_best_organism,
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
            after_population_selection=capture_best_organism,
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
        
        instructor_best = best_organisms.get(f"{experiment_name}_Instructor_Cycle_{cycle}")
        student_best = best_organisms.get(f"{experiment_name}_Student_Cycle_{cycle}")
        control_best = best_organisms.get(f"{experiment_name}_Control_Cycle_{cycle}")
        
       
        
        # Log results and compare the phases
        print("\n--- Results Summary ---")
        compare_results(instructor_ga, student_ga, control_ga, cycle)

def compare_results(instructor_ga, student_ga, control_ga, cycle):
    # Implement your logic to compare and log results from each phase for the current cycle
    # Placeholder for demonstration purposes
    print(f"Results for Cycle {cycle}:\nInstructor Best Fitness: {max(instructor_ga.fitness_scores)}\nStudent Best Fitness: {max(student_ga.fitness_scores)}\nControl Best Fitness: {max(control_ga.fitness_scores)}")

def capture_best_organism(ga_instance):
    best_index = ga_instance.fitness_scores.index(max(ga_instance.fitness_scores))
    best_organism = ga_instance.population[best_index]

    # Decode the best organism using the instance's method
    decoded_organism = ga_instance.decode_organism(best_organism, format=True)

    # Print the decoded organism and its fitness
    print(f"Decoded Best Organism: {decoded_organism}")
    print(f"Fitness: {ga_instance.fitness_scores[best_index]}")

    # Convert decoded organism into 3D coordinates
    x, y, z = 0, 0, 0
    positions = [(x, y, z)]  # Initial position
    for gene in decoded_organism:
        if gene in directions:
            dx, dy, dz = directions[gene]
            x += dx
            y += dy
            z += dz
            positions.append((x, y, z))

    # Store the 3D coordinates along with its fitness score
    best_organisms[ga_instance.experiment_name] = (positions, ga_instance.fitness_scores[best_index])

def plot_organism(organism_coordinates, title='Best Organism'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = [coord[0] for coord in organism_coordinates]
    ys = [coord[1] for coord in organism_coordinates]
    zs = [coord[2] for coord in organism_coordinates]

    # Plotting a line from each point to the next in the organism's path
    ax.plot(xs, ys, zs, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    genes = GENES
    experiment_name = input("Your_Experiment_Name: ")
    best_organisms = {}
    run_experiment(experiment_name, num_cycles, genes, protein_folding_fitness_function)
    
    # Plotting the best organisms from each cycle
    for experiment_name, (organism_coordinates, fitness) in best_organisms.items():
        print(organism_coordinates)
        plot_organism(organism_coordinates, title=f'{experiment_name} - Fitness: {fitness}')
