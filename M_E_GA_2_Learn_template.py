import random
from M_E_GA_Base_V2 import M_E_GA_Base

# Global settings and constants
GLOBAL_SEED = None
NUM_CYCLES = 1
MAX_GENERATIONS = 200
random.seed(GLOBAL_SEED)

# Genetic Algorithm Constants
MAX_LENGTH = 15000
GENES = ['0', '1']
MAX_INDIVIDUAL_LENGTH = 25

# Common configuration dictionary for GA phases
common_config = {
    'max_generations': MAX_GENERATIONS,
    'delimiters': False,
    'delimiter_space': 3,
    'logging': False,
    'generation_logging': True,
    'mutation_logging': False,
    'crossover_logging': False,
    'individual_logging': True
}

# Phase specific settings
phase_settings = {
    "instructor": {
        'mutation_prob': 0.065,
        'delimited_mutation_prob': 0.08,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.002,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.45,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.45,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 700,
        'num_parents': 150
    },
    "student": {
        'mutation_prob': 0.065,
        'delimited_mutation_prob': 0.08,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.002,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.50,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.45,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 700,
        'num_parents': 150
    },
    "nd_learner": {
        'mutation_prob': 0.065,
        'delimited_mutation_prob': 0.08,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.002,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.50,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.50,
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
    decoded_individual = ga_instance.decode_organism(encoded_individual)
    fitness_score = sum(1 ** i if gene == '1' else 0 for i, gene in enumerate(decoded_individual))
    penalty = (1.008 ** (MAX_LENGTH - len(decoded_individual)) if len(decoded_individual) < MAX_LENGTH else (
                len(decoded_individual) - MAX_LENGTH))
    update_best_organism(encoded_individual, fitness_score, verbose=True)
    return fitness_score - penalty


class ExperimentGA(M_E_GA_Base):
    def __init__(self, phase, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = phase

    def run_phase(self):
        print(f"Running {self.phase} phase with settings: {phase_settings[self.phase]}")
        self.run_algorithm()
        return self.get_best_organism_info(), self.encoding_manager.encodings

    def get_best_organism_info(self):
        best_fitness = max(self.fitness_scores)
        best_index = self.fitness_scores.index(best_fitness)
        best_organism = self.population[best_index]
        decoded_best_organism = self.decode_organism(best_organism, format=True)
        return {"fitness": best_fitness, "organism": decoded_best_organism}

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
