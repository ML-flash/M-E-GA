import random
from M_E_GA_Base_V2 import M_E_GA_Base
from concurrent.futures import ThreadPoolExecutor
import os
from M_E_GA_GP_fitness_template import MegaGPFitnessFunction  # Import the MegaGPFitnessFunction class

# Constants and configurations
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)
NUM_CYCLES = 1
MAX_GENERATIONS = 150
INPUT_SIZE = 9  # Set based on the expected number of input bits for MegaGP

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
        'mutation_prob': 0.30,
        'delimited_mutation_prob': 0.06,
        'open_mutation_prob': 0.009,
        'capture_mutation_prob': 0.003,
        'delimiter_insert_prob': 0.002,
        'crossover_prob': 0.50,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.50,
        'capture_gene_prob': 0.2,
        'max_individual_length': 40,
        'population_size': 700,
        'num_parents': 150
    },
    "student": {
        'mutation_prob': 0.30,
        'delimited_mutation_prob': 0.07,
        'open_mutation_prob': 0.009,
        'capture_mutation_prob': 0.003,
        'delimiter_insert_prob': 0.002,
        'crossover_prob': 0.50,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.40,
        'capture_gene_prob': 0.2,
        'max_individual_length': 40,
        'population_size': 700,
        'num_parents': 150
    },
    "nd_learner": {
        'mutation_prob': 0.05,
        'delimited_mutation_prob': 0.05,
        'open_mutation_prob': 0.00,
        'capture_mutation_prob': 0.00,
        'delimiter_insert_prob': 0.00,
        'crossover_prob': 0.90,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.5,
        'capture_gene_prob': 0.2,
        'max_individual_length': 40,
        'population_size': 700,
        'num_parents': 150
    }
}

# Initialize the MegaGPFitnessFunction with update function passed in
fitness_function = MegaGPFitnessFunction(INPUT_SIZE, update_best_func=update_best_organism)

class ExperimentGA(M_E_GA_Base):
    def __init__(self, phase, cycle, experiment_name, genes, fitness_function, **kwargs):
        super().__init__(genes, fitness_function, **kwargs)
        self.experiment_name = f"{experiment_name}_Phase_{phase}_Cycle_{cycle}"
        self.phase = phase

    def run_phase(self):
        print(f"Running {self.phase} phase with settings: {phase_settings[self.phase]}")
        self.run_algorithm()
        return self.get_best_organism_info()

    def get_best_organism_info(self):
        best_fitness = max(self.fitness_scores)
        best_index = self.fitness_scores.index(best_fitness)
        best_organism = self.population[best_index]
        decoded_best_organism = self.decode_organism(best_organism, format=True)
        return {"fitness": best_fitness, "organism": decoded_best_organism}

def run_experiment(experiment_name, num_cycles):
    for cycle in range(1, num_cycles + 1):
        print(f"\n--- Starting Experiment Cycle {cycle} ---")
        for phase in ['instructor', 'student', 'nd_learner']:
            ga = ExperimentGA(phase, cycle, experiment_name, genes=fitness_function.genes, fitness_function=fitness_function.compute, **common_config, **phase_settings[phase])
            results = ga.run_phase()
            print(f"{phase.capitalize()} Phase Results: Fitness = {results['fitness']}, Organism = {results['organism']}")

if __name__ == '__main__':
    experiment_name = input("Enter Experiment Name: ")
    run_experiment(experiment_name, NUM_CYCLES)
