import random
from M_E_GA_Base_V2 import M_E_GA_Base


# Global settings and constants
GLOBAL_SEED = None
NUM_CYCLES = 1
MAX_GENERATIONS = 200
random.seed(GLOBAL_SEED)

# Genetic Algorithm Constants
MAX_LENGTH = 200
GENES = ['0', '1']
MAX_INDIVIDUAL_LENGTH = 100

# Common configuration dictionary for GA phases
common_config = {
    'max_generations': MAX_GENERATIONS,
    'delimiters': False,
    'delimiter_space': 3,
    'logging': False,
    'generation_logging': False,
    'mutation_logging': False,
    'crossover_logging': False,
    'individual_logging': True
}

# Phase specific settings
phase_settings = {
    "instructor": {
        'mutation_prob': 0.10,
        'delimited_mutation_prob': 0.05,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.002,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.70,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.60,
        'capture_gene_prob': 0.1,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 700,
        'num_parents': 150
    },
    "student": {
        'mutation_prob': 0.05,
        'delimited_mutation_prob': 0.03,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.001,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.70,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.45,
        'capture_gene_prob': 0.14,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 700,
        'num_parents': 150
    },
    "nd_learner": {
        'mutation_prob': 0.02,
        'delimited_mutation_prob': 0.01,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.001,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.70,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.45,
        'capture_gene_prob': 0.15,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 700,
        'num_parents': 150
    }
}

best_organism = {
    "genome": None,
    "fitness": float('-inf')  # Initialize best organism with lowest possible fitness
}


def update_best_organism(current_genome, current_fitness, verbose=False):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")


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

    # Update the best organism (assuming this function is defined elsewhere)
    update_best_organism(encoded_individual, fitness_score, verbose=True)

    # Return the final fitness score after applying the penalty
    return fitness_score - penalty


class ExperimentGA(M_E_GA_Base):
    def __init__(self, phase, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = phase

    def run_phase(self):
        print(f"Running {self.phase} phase with settings: {phase_settings[self.phase]}")
        self.run_algorithm()


    def load_encodings(self, encodings):
        self.encoding_manager.integrate_uploaded_encodings(encodings, GENES)


def run_experiment(experiment_name, num_cycles):
    for cycle in range(1, num_cycles + 1):
        print(f"\n--- Starting Experiment Cycle {cycle} ---")

        # Run Instructor Phase
        instructor_ga = ExperimentGA('instructor', genes=GENES, fitness_function=leading_ones_fitness_function,
                                     **common_config, **phase_settings['instructor'])
        instructor_results, instructor_encodings = instructor_ga.run_phase()

        # Run Student Phase
        student_ga = ExperimentGA('student', genes=GENES, fitness_function=leading_ones_fitness_function,
                                  **common_config, **phase_settings['student'])
        student_ga.load_encodings(instructor_encodings)
        student_results, student_encodings = student_ga.run_phase()

        # Run ND Learner Phase
        nd_learner_ga = ExperimentGA('nd_learner', genes=GENES, fitness_function=leading_ones_fitness_function,
                                     **common_config, **phase_settings['nd_learner'])
        nd_learner_ga.load_encodings(student_encodings)
        nd_learner_results, _ = nd_learner_ga.run_phase()

        # Summarize results for the cycle
        print("\n--- Cycle Results Summary ---")
        print(f"Instructor Best: {instructor_results['fitness']}")
        print(f"Student Best: {student_results['fitness']}")
        print(f"ND Learner Best: {nd_learner_results['fitness']}")


if __name__ == '__main__':
    experiment_name = input("Enter Experiment Name: ")
    run_experiment(experiment_name, NUM_CYCLES)
