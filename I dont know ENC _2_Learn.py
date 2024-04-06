import random
from M_E_GA_Base_V2 import M_E_GA_Base
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GLOBAL_SEED =            None
NUM_CYCLES =              1
MAX_GENERATIONS =         800
random.seed(GLOBAL_SEED)

VOLUME = 5
NUM_ITEMS = 40
NUM_GROUPS = 4


MUTATION_PROB =           0.01
DELIMITED_MUTATION_PROB = 0.01
OPEN_MUTATION_PROB =      0.007
CAPTURE_MUTATION_PROB =   0.001
DELIMITER_INSERT_PROB =   0.004
CROSSOVER_PROB =          .70
ELITISM_RATIO =           0.6
BASE_GENE_PROB =          0.50
MAX_INDIVIDUAL_LENGTH =   400
POPULATION_SIZE =         700
NUM_PARENTS =             150
DELIMITER_SPACE =         3
DELIMITERS =              False


LOGGING =                 True
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



def create_items(num_items, num_groups, properties=('size', 'weight', 'density', 'value')):
    """
    Generates a list of items, each with its own properties, interactions, and reaction strength.

    Parameters:
    - num_items: Total number of items to create.
    - num_groups: Total number of distinct groups items can belong to.
    - properties: A list of property names that can be affected by interactions.

    Returns:
    A list of dictionaries, each representing an item with its properties, interactions, and reaction strength.
    """
    items = []

    for item_id in range(num_items):
        item = {
            'id': item_id,
            'group': random.randint(0, num_groups - 1),
            'properties': {
                'size': random.uniform(0.1, 1.0),
                'weight': random.uniform(0.1, 1.0),
                'density': random.uniform(0.1, 1.0),
                'value': random.uniform(0.1, 1.0),
            },
            'reaction_strength': random.uniform(0.5, 1.5),  # Define the reaction strength with a random value
            'interactions': []  # Initialize an empty list for interactions
        }

        # Determine the number of groups this item will interact with (up to the total number of groups)
        num_interactions = random.randint(0, num_groups - 1)

        # Randomly select the groups for interaction, ensuring no repeats and not including the item's own group
        interacting_groups = random.sample([group for group in range(num_groups) if group != item['group']], num_interactions)

        # Define interactions for each selected group
        for target_group in interacting_groups:
            interaction = {
                'target_group': target_group,
                'property': random.choice(properties),
                'direction': random.choice(['increase', 'decrease']),
                'magnitude': random.uniform(0.1, 0.5),  # Adjust magnitude range as needed
            }
            # Add the defined interaction to the item's interactions list
            item['interactions'].append(interaction)

        # Add the fully defined item to the items list
        items.append(item)

    return items


def can_add_item_to_sack(item, sack):
    # Example based on item size and sack's remaining capacity
    return (sack['current_capacity'] + item['properties']['size']) <= sack['max_capacity']


def collect_item(item, sack, verbose=False):
    sack['items'].append(item)  # Add item to sack
    sack['current_capacity'] += item['properties']['size']  # Update sack's capacity usage
    if verbose:
        print(f"Collected item {item['id']} with size {item['properties']['size']}. Current sack capacity: {sack['current_capacity']}/{sack['max_capacity']}")
    # Optionally, apply item's interactions immediately upon collection
    apply_interactions(item, sack['items'], verbose=verbose)


def apply_interactions(new_item, items_in_sack, verbose=False):
    for interaction in new_item['interactions']:
        for item in items_in_sack:
            if item['group'] == interaction['target_group']:
                affected_property = interaction['property']
                direction = interaction['direction']
                magnitude = interaction['magnitude'] * new_item['reaction_strength']
                change = magnitude if direction == 'increase' else -magnitude
                old_value = item['properties'][affected_property]
                item['properties'][affected_property] = max(0, old_value + change)
                if verbose:
                    print(f"Applying interaction from item {new_item['id']} to item {item['id']}: {affected_property} {'increased' if change > 0 else 'decreased'} from {old_value} to {item['properties'][affected_property]}")


def calculate_sack_value(sack, verbose=False):
    total_value = sum(item['properties']['value'] for item in sack['items'])
    # Apply penalties or adjustments based on sack constraints if necessary
    if verbose:
        print(f"Total sack value: {total_value}")
    return total_value


def problem_specific_fitness_function(encoded_individual, ga_instance, items, volume_bound=VOLUME, sack_capacity=100, verbose=False):
    decoded_individual = ga_instance.decode_organism(encoded_individual)
    fitness_score = 0
    x, y, z = 0, 0, 0  # Starting position
    visited_positions = set([(x, y, z)])  # Track visited positions to avoid self-collision
    sack = {'items': [], 'current_capacity': 0, 'max_capacity': sack_capacity}  # Initialize the sack

    for gene in decoded_individual:
        if gene in directions:
            dx, dy, dz = directions[gene]  # Get the direction vector
            new_pos = (x + dx, y + dy, z + dz)  # Calculate the new position

            # Verbose logging for the move
            if verbose:
                print(f"Moving from {(x, y, z)} to {new_pos}")

            # Check if the new position is valid
            if not (-volume_bound <= new_pos[0] <= volume_bound and
                    -volume_bound <= new_pos[1] <= volume_bound and
                    -volume_bound <= new_pos[2] <= volume_bound) or new_pos in visited_positions:
                if verbose:
                    print("Invalid move detected. Ending gene processing.")
                break  # Invalid move: outside volume or revisiting position

            # Check for item collection at the new position
            for item in items:
                if 'position' in item and item['position'] == new_pos and can_add_item_to_sack(item, sack):
                    collect_item(item, sack, verbose=verbose)  # Collect the item if it fits in the sack

            fitness_score += 1  # Reward for each valid move
            visited_positions.add(new_pos)  # Mark the new position as visited
            x, y, z = new_pos  # Update the current position

    fitness_score += calculate_sack_value(sack, verbose=verbose)  # Add value of collected items to the fitness score

    # Verbose logging for the final fitness score
    if verbose:
        print(f"Final fitness score: {fitness_score}")

    return fitness_score, {}


class ExperimentGA(M_E_GA_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items = items

    def fitness_function_wrapper(self, individual):
        # Correctly pass the individual to the problem-specific fitness function
        # and use 'self' to pass the current GA instance
        return problem_specific_fitness_function(individual, self, self.items)

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
            items=items,
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
            items=items,
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
            items=items,
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
    items = create_items(num_items= NUM_ITEMS, num_groups=NUM_GROUPS)
    for item in items:
        item['position'] = (random.randint(-VOLUME, VOLUME),
                            random.randint(-VOLUME, VOLUME),
                            random.randint(-VOLUME, VOLUME))

    GENES = ['R', 'L', 'U', 'D', 'F', 'B']
    directions = {'R': (1, 0, 0), 'L': (-1, 0, 0), 'U': (0, 1, 0), 'D': (0, -1, 0), 'F': (0, 0, 1), 'B': (0, 0, -1)}
    experiment_name = input("Your_Experiment_Name: ")
    best_organisms = {}
    run_experiment(experiment_name, NUM_CYCLES, GENES, lambda ind, ga_inst: problem_specific_fitness_function(ind, ga_inst, items))