import random
from M_E_GA2 import M_E_GA_Base
from M_E_GA_GP_fitness_template import MegaGPFitnessFunction

INPUT_SIZE = 10

# Constants and configurations
GLOBAL_SEED = 5278
random.seed(GLOBAL_SEED)
NUM_CYCLES = 5
MAX_GENERATIONS = 200
MAX_INDIVIDUAL_LENGTH = 45

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

# Common configuration dictionary for GA phases
common_config = {
    'max_generations': MAX_GENERATIONS,
    'delimiters': False,
    'delimiter_space': 3,
    'logging': True,
    'generation_logging': True,
    'mutation_logging': False,
    'crossover_logging': False,
    'individual_logging': False,
    'seed': GLOBAL_SEED
}

# Phase specific settings
phase_settings = {
    "instructor": {
        'mutation_prob': 0.15,
        'delimited_mutation_prob': 0.06,
        'open_mutation_prob': 0.004,
        'capture_mutation_prob': 0.009,
        'delimiter_insert_prob': 0.002,
        'crossover_prob': 0.50,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.40,
        'capture_gene_prob': 0.0,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 600,
        'num_parents': 200
    },
    "student": {
        'mutation_prob': 0.15,
        'delimited_mutation_prob': 0.06,
        'open_mutation_prob': 0.0,
        'capture_mutation_prob': 0.00,
        'delimiter_insert_prob': 0.00,
        'crossover_prob': 0.70,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0.40,
        'capture_gene_prob': 0.0,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 600,
        'num_parents': 200
    },
    "control": {
        'mutation_prob': 0.10,
        'delimited_mutation_prob': 0.05,
        'open_mutation_prob': 0.00,
        'capture_mutation_prob': 0.00,
        'delimiter_insert_prob': 0.00,
        'crossover_prob': 0.50,
        'elitism_ratio': 0.6,
        'base_gene_prob': 0,
        'capture_gene_prob': 0,
        'max_individual_length': MAX_INDIVIDUAL_LENGTH,
        'population_size': 600,
        'num_parents': 200
    }
}

# Initialize the fitness function with update function passed in
fitness_function = MegaGPFitnessFunction(INPUT_SIZE, update_best_func=update_best_organism)

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
        self.encoding_manager.integrate_uploaded_encodings(encodings, fitness_function.genes, verbose=True)

def run_experiment(experiment_name, num_cycles=NUM_CYCLES):
    phases = ["instructor", "student", "control"]
    overall_best = []
    encodings = None

    for cycle in range(1, num_cycles + 1):
        for phase in phases:
            ga = ExperimentGA(phase, cycle, experiment_name, genes=fitness_function.genes,
                              fitness_function=fitness_function.compute, **common_config, **phase_settings[phase])
            if phase == "student" and encodings:
                ga.load_encodings(encodings)
            results, new_encodings = ga.run_phase()
            if phase == "instructor":
                encodings = new_encodings  # Save encodings from the instructor phase
            overall_best.append((phase, results['fitness'], results['organism']))
            print(f"\n--- {phase.capitalize()} Phase Results Summary ---")
            print(f"{phase.capitalize()} Best: {results['fitness']}")

    print("\n--- Overall Best Results Summary ---")
    for phase, fitness, organism in overall_best:
        print(f"{phase.capitalize()} Phase Best Fitness: {fitness}")
        print(f"{phase.capitalize()} Phase Best Organism: {organism}\n")

if __name__ == '__main__':
    experiment_name = input("Enter Experiment Name: ")
    run_experiment(experiment_name)