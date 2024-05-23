import random
from M_E_GA_Base_V2 import M_E_GA_Base
from concurrent.futures import ThreadPoolExecutor
import os
from M_E_GA_fitness_funcs import LeadingOnesFitness  # Import the modified fitness function class

MAX_LENGTH = 300
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)

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

def evaluate_individual(individual, encoding_manager):
    return encoding_manager.fitness_function(individual)

def evaluate_population(population, encoding_manager, num_threads=None):
    if num_threads is None:
        num_threads = os.cpu_count()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda ind: evaluate_individual(ind, encoding_manager), population))
    return results

# Common configuration dictionary for GA phases
common_config = {
    'max_generations': 10,
    'delimiters': False,
    'delimiter_space': 3,
    'logging': True,
    'generation_logging': False,
    'mutation_logging': False,
    'crossover_logging': False,
    'individual_logging': True,
    'seed': GLOBAL_SEED
}

# Phase settings template
phase_settings_template = {
    'instructor': {
        'mutation_prob': 0.10,
        'delimited_mutation_prob': 0.07,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.001,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.80,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.60,
        'capture_gene_prob': 0.1,
        'max_individual_length': 100,
        'population_size': 700,
        'num_parents': 150
    },
    'student': {
        'mutation_prob': 0.10,
        'delimited_mutation_prob': 0.07,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.002,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.80,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.35,
        'capture_gene_prob': 0.1,
        'max_individual_length': 100,
        'population_size': 700,
        'num_parents': 150
    },
    'nd_learner': {
        'mutation_prob': 0.02,
        'delimited_mutation_prob': 0.01,
        'open_mutation_prob': 0.007,
        'capture_mutation_prob': 0.001,
        'delimiter_insert_prob': 0.004,
        'crossover_prob': 0.80,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.40,
        'capture_gene_prob': 0.15,
        'max_individual_length': 100,
        'population_size': 700,
        'num_parents': 150
    }
}

# Initialize the fitness function with update function passed in
fitness_function = LeadingOnesFitness(max_length=MAX_LENGTH, update_best_func=update_best_organism)

class ExperimentGA(M_E_GA_Base):
    def __init__(self, phase, cycle, experiment_name, *args, **kwargs):
        # Formulate the specific experiment name including phase and cycle
        modified_experiment_name = f"{experiment_name}_Phase_{phase}_Cycle_{cycle}"
        super().__init__(*args, **kwargs, experiment_name=modified_experiment_name)
        self.phase = phase

    def run_phase(self):
        print(f"Running {self.phase} phase with settings: {phase_settings_template[self.phase]}")
        self.run_algorithm()
        return self.get_best_organism_info(), self.encoding_manager.encodings

    def run_algorithm(self):
        for generation in range(self.max_generations):
            self.evaluate_population()
            self.select_parents()
            self.perform_crossover_and_mutation()
            self.update_population()
            if self.logging and self.generation_logging:
                print(f"Generation {generation}: Best Fitness {max(self.fitness_scores)}")

    def evaluate_population(self):
        self.fitness_scores = evaluate_population(self.population, self.encoding_manager)

    def get_best_organism_info(self):
        best_fitness = max(self.fitness_scores)
        best_index = self.fitness_scores.index(best_fitness)
        best_organism = self.population[best_index]
        decoded_best_organism = self.decode_organism(best_organism, format=True)
        return {"fitness": best_fitness, "organism": decoded_best_organism}

    def load_encodings(self, encodings):
        self.encoding_manager.integrate_uploaded_encodings(encodings, fitness_function.genes)


def run_experiment(experiment_name, num_cycles, phase_sequence):
    for cycle in range(1, num_cycles + 1):
        print(f"\n--- Starting Experiment Cycle {cycle} ---")
        previous_encodings = None

        for phase in phase_sequence:
            ga = ExperimentGA(
                phase, cycle, experiment_name,
                genes=fitness_function.genes,
                fitness_function=fitness_function.compute,
                **common_config, **phase_settings_template[phase]
            )
            if previous_encodings is not None:
                ga.load_encodings(previous_encodings)
            results, encodings = ga.run_phase()
            previous_encodings = encodings

            # Summarize results for the phase
            print(f"{phase.capitalize()} Phase Best Fitness: {results['fitness']}")

if __name__ == '__main__':
    experiment_name = input("Enter Experiment Name: ")
    phase_sequence = ['instructor', 'student', 'nd_learner']  # Example phase sequence
    run_experiment(experiment_name, NUM_CYCLES, phase_sequence)
