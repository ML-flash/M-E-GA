import random
import math
import numpy as np
from M_E_GA_Base_V2 import M_E_GA_Base
import matplotlib.pyplot as plt
import cellpylib as cpl


GLOBAL_SEED =            None
NUM_CYCLES =              1
MAX_GENERATIONS =         200
random.seed(GLOBAL_SEED)

STEPS = 100
RULE = 110


MUTATION_PROB =           0.03
DELIMITED_MUTATION_PROB = 0.03
OPEN_MUTATION_PROB =      0.008
CAPTURE_MUTATION_PROB =   0.0006
DELIMITER_INSERT_PROB =   0.005
CROSSOVER_PROB =          .70
ELITISM_RATIO =           0.06
BASE_GENE_PROB =          0.80
MAX_INDIVIDUAL_LENGTH =   80
POPULATION_SIZE =         500
NUM_PARENTS =             100
DELIMITER_SPACE =         3
DELIMITERS =              False


LOGGING =                 True
GENERATION_LOGGING =      False
MUTATION_LOGGING =        True
CROSSOVER_LOGGING =       False
INDIVIDUAL_LOGGING =      False

#Student Settings

S_MUTATION_PROB =           0.03
S_DELIMITED_MUTATION_PROB = 0.03
S_OPEN_MUTATION_PROB =      0.004
S_CAPTURE_MUTATION_PROB =   0.001
S_DELIMITER_INSERT_PROB =   0.004
S_CROSSOVER_PROB =          .70
S_ELITISM_RATIO =           0.06
S_BASE_GENE_PROB =          0.50
S_MAX_INDIVIDUAL_LENGTH =   200
S_POPULATION_SIZE =         500
S_NUM_PARENTS =             100
S_DELIMITER_SPACE =         3
S_DELIMITERS =              False

# ND Learner settings
ND_MUTATION_PROB =           0.03
ND_DELIMITED_MUTATION_PROB = 0.03
ND_OPEN_MUTATION_PROB =      0.007
ND_CAPTURE_MUTATION_PROB =   0.0008
ND_DELIMITER_INSERT_PROB =   0.004
ND_CROSSOVER_PROB =          .70
ND_ELITISM_RATIO =           0.06
ND_BASE_GENE_PROB =          0.50
ND_MAX_INDIVIDUAL_LENGTH =   200
ND_POPULATION_SIZE =         500
ND_NUM_PARENTS =             100
ND_DELIMITER_SPACE =         3
ND_DELIMITERS =              False


best_organism = {
    "genome": None,
    "fitness": float('-inf')  # Start with negative infinity to ensure any valid organism will surpass it
}

def update_best_organism(current_genome, current_fitness, verbose = False):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")

# Helper functions for logic operations
def logic_and(inputs):
    return int(inputs[0] and inputs[1])

def logic_or(inputs):
    return int(inputs[0] or inputs[1])

def logic_not(inputs):
    return int(not inputs[0])


def evaluate_and_gate_noise_free(config, rule, logic_func, steps=STEPS, reward=1, penalty_rate=0.05, verbose=False):
    fitness_score = 0  # Initialize fitness score
    center_index = len(config) // 2  # Define the center index for the 'and' operation output

    # Loop through all possible input combinations for the 'and' operation
    for inputs in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        # Create a test configuration with the current inputs placed at the center
        test_config = np.array(config)
        test_config[center_index - 1: center_index + 1] = inputs

        # Run the cellular automaton simulation
        final_config = simulate_ca_np(rule, test_config, steps, verbose)

        # Determine the expected result from the 'and' operation
        expected = logic_func(inputs)
        noise_counter = 0  # Reset the noise counter for each set of inputs

        # Evaluate each cell in the final configuration
        for idx, cell in enumerate(final_config[-1]):
            if cell == expected:
                if idx == center_index:
                    # Reward for correct output at the center cell
                    if verbose and cell == expected:
                        print(f"Correct output for {inputs} at center.")
                    fitness_score += reward
                else:
                    # Apply a distance-based noise penalty
                    distance = abs(center_index - idx)
                    noise_penalty = penalty_rate / (distance + 1)  # Adding 1 to prevent division by zero
                    fitness_score -= noise_penalty
                    noise_counter += noise_penalty  # Track total noise penalty for verbose output

        if verbose:
            print(f"Input {inputs}: Applied noise penalty {noise_counter}, Current fitness score: {fitness_score}")

    # Normalize the fitness score and ensure it is not negative
    fitness_score = max(0, fitness_score) / 4
    return fitness_score




def simulate_ca_np(rule, initial_config, steps=STEPS, verbose=False):
    config = np.array(initial_config, dtype=int)
    result_matrix = np.zeros((steps + 1, len(config)), dtype=int)
    result_matrix[0] = config

    for step in range(1, steps + 1):
        left = np.roll(config, 1)
        right = np.roll(config, -1)
        neighborhood = (left << 2) | (config << 1) | right
        config = (rule >> neighborhood) & 1
        result_matrix[step] = config

    if verbose:
        plt.figure(figsize=(10, 5))
        plt.imshow(result_matrix, cmap='gray_r', interpolation='nearest')  # Using 'gray_r' for reversed grayscale
        plt.title(f'Cellular Automaton Rule {rule} Evolution')
        plt.xlabel('Cell Index')
        plt.ylabel('Step')
        plt.colorbar(label='Cell State')
        plt.show()

    return result_matrix


def evaluate_logic_operation_np(config, rule, logic_funcs, output_indices, lane_width=3, steps=STEPS, penalty_rate=0.01, verbose=False):
    total_correct = 0
    penalties = 0
    input_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    total_possible = len(input_combinations) * len(logic_funcs)

    for inputs in input_combinations:
        test_config = np.pad(config, (lane_width, lane_width), constant_values=(0, 0))  # pad to handle boundary conditions
        mid_index = len(test_config) // 2
        test_config[mid_index - 1: mid_index + 1] = inputs
        final_config = simulate_ca_np(rule, test_config, steps, verbose)

        for func, idx in zip(logic_funcs, output_indices):
            # Adjust index for padding
            idx += lane_width
            if idx >= len(final_config[-1]):
                continue  # skip if index is out of bounds
            expected = func(inputs)
            result = final_config[-1][idx]
            if result == expected:
                total_correct += 1

            # Check for noise in the lane
            start_idx = max(0, idx - lane_width)
            end_idx = min(len(final_config[-1]), idx + lane_width + 1)
            expected_state = 1 - result
            lane_noise = np.sum(final_config[-1][start_idx:end_idx] != expected_state) - (final_config[-1][idx] != expected_state)
            penalties += (lane_noise * 2)
            if verbose and lane_noise > 0:
                print(f"Noise detected near output index {idx} for input ({inputs}).")

    correct_ratio = total_correct / total_possible
    fitness_score = correct_ratio - (penalties * penalty_rate)
    return max(0, fitness_score)

def problem_specific_fitness_function(encoded_individual, ga_instance, required_length=MAX_INDIVIDUAL_LENGTH, verbose=False):
    decoded_individual = ga_instance.decode_organism(encoded_individual, format=True)
    decoded_individual = np.array([int(gene) for gene in decoded_individual], dtype=int)

    if len(decoded_individual) < required_length:
        decoded_individual = np.pad(decoded_individual, (0, required_length - len(decoded_individual)), 'constant')

    # Evaluate using the new noise-free and gate focus function
    fitness_score = evaluate_and_gate_noise_free(decoded_individual, RULE, logic_and, verbose=verbose)

    update_best_organism(encoded_individual, fitness_score, verbose = True)
    return max(0, fitness_score), {}

# Gene set and configuration for the GA
GENES = ['0', '1']  # Binary gene set for initial conditions of CA


class ExperimentGA(M_E_GA_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fitness_function_wrapper(self, individual, **kwargs):
        # Pass any keyword arguments to the fitness function
        return problem_specific_fitness_function(individual, self, **kwargs)

    def instructor_phase(self):
        self.run_algorithm()
        return self.encoding_manager.encodings

    def student_phase(self, instructor_encodings):
        self.encoding_manager.integrate_uploaded_encodings(instructor_encodings, GENES)
        self.run_algorithm()
        return self.encoding_manager.encodings

    def nd_learner_phase(self, student_encodings):
        self.encoding_manager.integrate_uploaded_encodings(student_encodings, GENES)
        self.run_algorithm()


def run_experiment(experiment_name, num_cycles, genes, fitness_function):
    for cycle in range(1, num_cycles + 1):
        print(f"\n--- Starting Experiment Cycle: {cycle} ---")

        instructor_ga = ExperimentGA(
            genes=GENES,
            fitness_function=fitness_function,
            after_population_selection=capture_best_organism,
            mutation_prob=MUTATION_PROB,
            delimited_mutation_prob=DELIMITED_MUTATION_PROB,
            open_mutation_prob=OPEN_MUTATION_PROB,
            capture_mutation_prob=CAPTURE_MUTATION_PROB,
            delimiter_insert_prob=DELIMITER_INSERT_PROB,
            crossover_prob=CROSSOVER_PROB,
            elitism_ratio=ELITISM_RATIO,
            base_gene_prob=BASE_GENE_PROB,
            max_individual_length=MAX_INDIVIDUAL_LENGTH,
            population_size=POPULATION_SIZE,
            num_parents=NUM_PARENTS,
            max_generations=MAX_GENERATIONS,
            delimiters=DELIMITERS,
            delimiter_space=DELIMITER_SPACE,
            logging=LOGGING,
            generation_logging=GENERATION_LOGGING,
            mutation_logging=MUTATION_LOGGING,
            crossover_logging=CROSSOVER_LOGGING,
            individual_logging=INDIVIDUAL_LOGGING,
            experiment_name=f"{experiment_name}_Instructor_Cycle_{cycle}",
            seed=GLOBAL_SEED
        )
        instructor_encodings = instructor_ga.instructor_phase()

        student_ga = ExperimentGA(
            genes=GENES,
            fitness_function=fitness_function,
            after_population_selection=capture_best_organism,
            mutation_prob=S_MUTATION_PROB,
            delimited_mutation_prob=S_DELIMITED_MUTATION_PROB,
            open_mutation_prob=S_OPEN_MUTATION_PROB,
            capture_mutation_prob=S_CAPTURE_MUTATION_PROB,
            delimiter_insert_prob=S_DELIMITER_INSERT_PROB,
            crossover_prob=S_CROSSOVER_PROB,
            elitism_ratio=S_ELITISM_RATIO,
            base_gene_prob=S_BASE_GENE_PROB,
            max_individual_length=S_MAX_INDIVIDUAL_LENGTH,
            population_size=S_POPULATION_SIZE,
            num_parents=S_NUM_PARENTS,
            max_generations=MAX_GENERATIONS,
            delimiters=S_DELIMITERS,
            delimiter_space=S_DELIMITER_SPACE,
            logging=LOGGING,
            generation_logging=GENERATION_LOGGING,
            mutation_logging=MUTATION_LOGGING,
            crossover_logging=CROSSOVER_LOGGING,
            individual_logging=INDIVIDUAL_LOGGING,
            experiment_name=f"{experiment_name}_Student_Cycle_{cycle}",
            seed=GLOBAL_SEED,
            )

        student_encodings = student_ga.student_phase(instructor_encodings)

        nd_learner_ga = ExperimentGA(
            genes=GENES,
            fitness_function=fitness_function,
            after_population_selection=capture_best_organism,
            mutation_prob=ND_MUTATION_PROB,
            delimited_mutation_prob=ND_DELIMITED_MUTATION_PROB,
            open_mutation_prob=ND_OPEN_MUTATION_PROB,
            capture_mutation_prob=ND_CAPTURE_MUTATION_PROB,
            delimiter_insert_prob=ND_DELIMITER_INSERT_PROB,
            crossover_prob=ND_CROSSOVER_PROB,
            elitism_ratio=ND_ELITISM_RATIO,
            base_gene_prob=ND_BASE_GENE_PROB,
            max_individual_length=ND_MAX_INDIVIDUAL_LENGTH,
            population_size=ND_POPULATION_SIZE,
            num_parents=ND_NUM_PARENTS,
            max_generations=MAX_GENERATIONS,
            delimiters=ND_DELIMITERS,
            delimiter_space=ND_DELIMITER_SPACE,
            logging=LOGGING,
            generation_logging=GENERATION_LOGGING,
            mutation_logging=MUTATION_LOGGING,
            crossover_logging=CROSSOVER_LOGGING,
            individual_logging=INDIVIDUAL_LOGGING,
            experiment_name=f"{experiment_name}_ND_Learner_Cycle_{cycle}",
            seed=GLOBAL_SEED,
             )
        nd_learner_ga.nd_learner_phase(student_encodings)

        instructor_best = best_organisms.get(f"{experiment_name}_Instructor_Cycle_{cycle}")
        student_best = best_organisms.get(f"{experiment_name}_Student_Cycle_{cycle}")
        nd_learner_best = best_organisms.get(f"{experiment_name}_Nd_Learner_Cycle_{cycle}")

        # Log results and compare the phases
        print("\n--- Results Summary ---")
        compare_results(nd_learner_ga)


def compare_results( ga):
    global best_organism
    _ = ga.fitness_function_wrapper(best_organism["genome"], verbose=True)


def capture_best_organism(ga_instance):
    best_index = ga_instance.fitness_scores.index(max(ga_instance.fitness_scores))
    best_organism = ga_instance.population[best_index]

    # Decode the best organism using the instance's method
    decoded_organism = ga_instance.decode_organism(best_organism, format=False)



if __name__ == '__main__':
    experiment_name = input("Your_Experiment_Name: ")
    best_organisms = {}
    run_experiment(experiment_name, NUM_CYCLES, GENES, lambda ind, ga_inst: problem_specific_fitness_function(ind, ga_inst, required_length=100, verbose=False))

