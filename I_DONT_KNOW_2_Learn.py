import random
from M_E_GA_Base_V2 import M_E_GA_Base
from concurrent.futures import ThreadPoolExecutor
import os
from I_Dont_Know_FF import IDontKnow

# Constants and configurations
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)
NUM_CYCLES = 1
MAX_GENERATIONS = 150
MAX_INDIVIDUAL_LENGTH = 40

VOLUME = 5
NUM_ITEMS = 1000
NUM_GROUPS = 100

best_organism = {
    "genome": None,
    "fitness": float('-inf')
}

def update_best_organism(current_genome, current_fitness, verbose=False):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")

# Common configuration dictionary for GA phases
common_config = {
    'max_generations': MAX_GENERATIONS,
    'delimiters': False,
    'delimiter_space': 3,
    'logging': False,
    'generation_logging': False,
    'mutation_logging': False,
    'crossover_logging': False,
    'individual_logging': True,
    'seed': GLOBAL_SEED
}

# Phase specific settings
phase_settings = {
    "instructor": {
        'mutation_prob': 0.10,
        'delimited_mutation_prob': 0.05,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.002,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.50,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.45,
        'capture_gene_prob': 0.2,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 700,
        'num_parents': 150
    },
    "student": {
        'mutation_prob': 0.10,
        'delimited_mutation_prob': 0.05,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.002,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.50,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.45,
        'capture_gene_prob': 0.2,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 700,
        'num_parents': 150
    },
    "nd_learner": {
        'mutation_prob': 0.05,
        'delimited_mutation_prob': 0.05,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.002,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.50,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.45,
        'capture_gene_prob': 0.2,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 700,
        'num_parents': 150
    }
}

# Initialize the fitness function with update function passed in
fitness_function = IDontKnow(volume = VOLUME, num_items= NUM_ITEMS, num_groups= NUM_GROUPS, update_best_func=update_best_organism)

class ExperimentGA(M_E_GA_Base):
    def __init__(self, phase, cycle, experiment_name, *args, **kwargs):
        # Formulate the specific experiment name including phase and cycle
        modified_experiment_name = f"{experiment_name}_Phase_{phase}_Cycle_{cycle}"
        super().__init__(*args, **kwargs, experiment_name=modified_experiment_name)
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
        self.encoding_manager.integrate_uploaded_encodings(encodings, fitness_function.genes)


def run_experiment(experiment_name, num_cycles):
    for cycle in range(1, num_cycles + 1):
        print(f"\n--- Starting Experiment Cycle {cycle} ---")

        # Run Instructor Phase
        instructor_ga = ExperimentGA('instructor', cycle, experiment_name, genes=fitness_function.genes, fitness_function=fitness_function.compute, **common_config, **phase_settings['instructor'])
        instructor_results, instructor_encodings = instructor_ga.run_phase()

        # Run Student Phase
        student_ga = ExperimentGA('student', cycle, experiment_name, genes=fitness_function.genes, fitness_function=fitness_function.compute, **common_config, **phase_settings['student'])
        student_ga.load_encodings(instructor_encodings)
        student_results, student_encodings = student_ga.run_phase()

        # Run ND Learner Phase
        nd_learner_ga = ExperimentGA('nd_learner', cycle, experiment_name, genes=fitness_function.genes, fitness_function=fitness_function.compute, **common_config, **phase_settings['nd_learner'])
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
