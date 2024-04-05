import random
from M_E_GA_Base_V2 import M_E_GA_Base
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GLOBAL_SEED =            None
NUM_CYCLES =              1
MAX_GENERATIONS =         800
random.seed(GLOBAL_SEED)


MUTATION_PROB =           0.01
DELIMITED_MUTATION_PROB = 0.01
OPEN_MUTATION_PROB =      0.007
CAPTURE_MUTATION_PROB =   0.001
DELIMITER_INSERT_PROB =   0.0004
CROSSOVER_PROB =          .70
ELITISM_RATIO =           0.6
BASE_GENE_PROB =          0.50
MAX_INDIVIDUAL_LENGTH =   400
POPULATION_SIZE =         700
NUM_PARENTS =             150
DELIMITER_SPACE =         3
DELIMITERS =              False


LOGGING =                 False
GENERATION_LOGGING =      True
MUTATION_LOGGING =        False
CROSSOVER_LOGGING =       False
INDIVIDUAL_LOGGING =      True

#Student Settings

S_MUTATION_PROB =           0.01
S_DELIMITED_MUTATION_PROB = 0.01
S_OPEN_MUTATION_PROB =      0.004
S_CAPTURE_MUTATION_PROB =   0.001
S_DELIMITER_INSERT_PROB =   0.004
S_CROSSOVER_PROB =          .90
S_ELITISM_RATIO =           0.6
S_BASE_GENE_PROB =          0.50
S_MAX_INDIVIDUAL_LENGTH =   400
S_POPULATION_SIZE =         700
S_NUM_PARENTS =             150
S_DELIMITER_SPACE =         3
S_DELIMITERS =              False

# ND Learner settings
ND_MUTATION_PROB =           0.001
ND_DELIMITED_MUTATION_PROB = 0.001
ND_OPEN_MUTATION_PROB =      0.007
ND_CAPTURE_MUTATION_PROB =   0.001
ND_DELIMITER_INSERT_PROB =   0.004
ND_CROSSOVER_PROB =          .90
ND_ELITISM_RATIO =           0.70
ND_BASE_GENE_PROB =          0.50
ND_MAX_INDIVIDUAL_LENGTH =   400
ND_POPULATION_SIZE =         700
ND_NUM_PARENTS =             150
ND_DELIMITER_SPACE =         3
ND_DELIMITERS =              False


GENES = ['R', 'L', 'U', 'D', 'F', 'B']
directions = {'R': (1, 0, 0), 'L': (-1, 0, 0), 'U': (0, 1, 0), 'D': (0, -1, 0), 'F': (0, 0, 1), 'B': (0, 0, -1)}


def problem_specific_fitness_function(encoded_individual, ga_instance):
    decoded_individual = ga_instance.decode_organism(encoded_individual)
    # Use ga_instance to decode the individual
    encoded_length = len(encoded_individual)

    # Initialize encoded_length to the length of the decoded_individual
    decoded_length = len(decoded_individual)
    fitness_score = 0
    x, y, z = 0, 0, 0
    volume_bound = 5  # Define the volume boundary
    visited_positions = set([(x, y, z)])  # Initialize with the starting position

    for gene in decoded_individual:
        if gene == 'Start':
            encoded_length -= 1
        elif gene == 'End':
            encoded_length -= 1
        if gene in directions:
            dx, dy, dz = directions[gene]
            new_pos = (x + dx, y + dy, z + dz)

            # Check if the new position is outside the defined volume or already visited
            if not (-volume_bound <= new_pos[0] <= volume_bound and
                    -volume_bound <= new_pos[1] <= volume_bound and
                    -volume_bound <= new_pos[2] <= volume_bound) or new_pos in visited_positions:
                break  # Stop evaluation and apply the length penalty

            # Valid move within the volume, increase fitness score and update visited positions
            fitness_score += 1
            visited_positions.add(new_pos)
            x, y, z = new_pos  # Update the current position

    return fitness_score, {}


class ExperimentGA(M_E_GA_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fitness_function_wrapper(self, individual):
        # Correctly pass the individual to the problem-specific fitness function
        # and use 'self' to pass the current GA instance
        return problem_specific_fitness_function(individual, self)

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
            seed=GLOBAL_SEED,
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
        compare_results(instructor_ga, student_ga, nd_learner_ga, cycle)


def compare_results(instructor_ga, student_ga, control_ga, cycle):
    # Implement your logic to compare and log results from each phase for the current cycle
    # Placeholder for demonstration purposes
    print(
        f"Results for Cycle {cycle}:\nInstructor Best Fitness: {max(instructor_ga.fitness_scores)}"
        f"\nStudent Best Fitness: {max(student_ga.fitness_scores)}\n Nd Learner Best Fitness: "
        f"{max(control_ga.fitness_scores)}")


def capture_best_organism(ga_instance):
    best_index = ga_instance.fitness_scores.index(max(ga_instance.fitness_scores))
    best_organism = ga_instance.population[best_index]

    # Decode the best organism using the instance's method
    decoded_organism = ga_instance.decode_organism(best_organism, format=False)



if __name__ == '__main__':
    # Initialize global variables for tasks, jobs, and machines
    # Initialize global variables for tasks, jobs, and machines
    GENES = ['R', 'L', 'U', 'D', 'F', 'B']
    directions = {'R': (1, 0, 0), 'L': (-1, 0, 0), 'U': (0, 1, 0), 'D': (0, -1, 0), 'F': (0, 0, 1), 'B': (0, 0, -1)}
    experiment_name = input("Your_Experiment_Name: ")
    best_organisms = {}
    run_experiment(experiment_name, NUM_CYCLES, GENES, problem_specific_fitness_function)
