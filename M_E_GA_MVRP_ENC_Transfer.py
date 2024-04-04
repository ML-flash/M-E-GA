import random
from M_E_GA_Base_V2 import M_E_GA_Base
import math

GGLOBAL_SEED =            None
NUM_CYCLES =              5
MAX_GENERATIONS =        600
random.seed(GLOBAL_SEED)


MUTATION_PROB =           0.015
DELIMITED_MUTATION_PROB = 0.015
OPEN_MUTATION_PROB =      0.007
CAPTURE_MUTATION_PROB =   0.001
DELIMITER_INSERT_PROB =   0.004
CROSSOVER_PROB =          .90
ELITISM_RATIO =           0.6
BASE_GENE_PROB =          0.35
MAX_INDIVIDUAL_LENGTH =   400
POPULATION_SIZE =         600
NUM_PARENTS =             100
DELIMITER_SPACE =         3
DELIMITERS =              False


LOGGING =                 True
GENERATION_LOGGING =      True
MUTATION_LOGGING =        False
CROSSOVER_LOGGING =       False
INDIVIDUAL_LOGGING =      True


NUM_VEHICLES = 2000
NUM_SUPPLIERS = 6000
DEPOT_PENALTY = -100
COST_LIMIT = 1500


best_organism = {
    "genome": None,
    "fitness": float('-inf')  # Start with negative infinity to ensure any valid organism will surpass it
}

distance_cache = {}


def length_penalty(decoded_length, encoded_length):
    return -2 * (decoded_length - encoded_length)



def calculate_distance_cached(point1, point2):
    # Sort points to ensure the cache key is consistent
    sorted_points = tuple(sorted([point1, point2]))
    if sorted_points in distance_cache:
        return distance_cache[sorted_points]

    distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    distance_cache[sorted_points] = distance
    return distance


def update_best_organism(current_genome, current_fitness, verbose = False):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")


# Function to generate vehicles
def generate_vehicles(num_vehicles, capacity_range, max_distance_range, cost_per_distance_range):
    return {
        f"V{i}": {
            "capacity": random.randint(*capacity_range),
            "max_distance": random.randint(*max_distance_range),
            "cost_per_distance": random.uniform(*cost_per_distance_range)
        }
        for i in range(1, num_vehicles + 1)
    }

# Function to generate suppliers
def generate_suppliers(num_suppliers, demand_range, time_window_ranges):
    return {
        f"S{i}": {
            "demand": 0,
            "time_window": (0, 0),
            "location": (0, 0)
        } if i == 0 else {
            "demand": random.randint(*demand_range),
            "time_window": (random.randint(*random.choice(time_window_ranges)),
                            random.randint(*random.choice(time_window_ranges))),
            "location": (random.uniform(0, 100), random.uniform(0, 100))
        }
        for i in range(0, num_suppliers + 1)
    }


# Problem-specific fitness function
def problem_specific_fitness_function(encoded_genome, ga_instance, vehicles, suppliers, depot_location=(0, 0),
                                      penalty_per_unit_time=10, penalty_per_unit_demand_excess=20,
                                      resupply_cost=50, base_reset_cost=5, cost_per_unit_distance=0.1,
                                      service_time=1, correct_order_reward=0.5, max_penalty_ratio=0.5,
                                      depot_return_reward=.1, distance_penalty_factor=0.01, DEPOT_PENALTY=-10,  # Example soft cost limit
                                      cost_limit_penalty_steepness=0.02):  # Controls penalty growth rate
    decoded_genome = ga_instance.decode_organism(encoded_genome)
    total_cost = 0
    total_reward = 0
    current_location = depot_location
    vehicle_distance = {}
    vehicle_load = {}
    visited_suppliers = set()
    depot_penalty_multiplier = 1  # Initialize multiplier for consecutive S0 penalty
    depot_reward_multiplier = 2

    last_vehicle = None
    returned_to_depot = False
    previous_gene_was_s0 = False  # Track consecutive S0s

    # Function to calculate the cost overrun penalty
    def cost_overrun_penalty(overrun):
        return 1 / (1 + math.exp(-cost_limit_penalty_steepness * (overrun - COST_LIMIT/2)))

    for i, gene in enumerate(decoded_genome):
        if gene in ['Start', 'End']:
            continue

        if gene == "S0":
            if previous_gene_was_s0:
                total_cost += DEPOT_PENALTY * depot_penalty_multiplier ** 0.5
                depot_penalty_multiplier += 1
            else:
                depot_penalty_multiplier = 1
            if last_vehicle:
                distance_to_depot = calculate_distance_cached(current_location, depot_location)
                total_cost += distance_to_depot * vehicles[last_vehicle]['cost_per_distance']
                if vehicle_distance[last_vehicle] <= vehicles[last_vehicle]['max_distance']:
                    total_reward += depot_return_reward * (vehicle_load[last_vehicle] * depot_reward_multiplier)
                vehicle_load[last_vehicle] = 0
                vehicle_distance[last_vehicle] = 0
                returned_to_depot = True
            current_location = depot_location
            previous_gene_was_s0 = True

        elif gene in vehicles:
            previous_gene_was_s0 = False
            if last_vehicle and not returned_to_depot:
                distance_to_depot = calculate_distance_cached(current_location, depot_location)
                distance_excess = distance_to_depot - vehicles[last_vehicle]['max_distance']
                if distance_excess > 0:
                    total_cost += distance_excess * vehicles[last_vehicle]['cost_per_distance'] * 0.5
            last_vehicle = gene
            returned_to_depot = False
            current_location = depot_location
            vehicle_distance[last_vehicle] = 0
            vehicle_load[last_vehicle] = 0

        elif gene in suppliers and last_vehicle:
            previous_gene_was_s0 = False
            supplier = suppliers[gene]
            distance = calculate_distance_cached(current_location, supplier['location'])
            if vehicle_distance[last_vehicle] + distance <= vehicles[last_vehicle]['max_distance']:
                vehicle_distance[last_vehicle] += distance
                if gene not in visited_suppliers:
                    visited_suppliers.add(gene)
                    vehicle_load[last_vehicle] += supplier['demand']
                    if vehicle_load[last_vehicle] > vehicles[last_vehicle]['capacity']:
                        total_cost += penalty_per_unit_demand_excess * (vehicle_load[last_vehicle] - vehicles[last_vehicle]['capacity'])
                        vehicle_load[last_vehicle] = vehicles[last_vehicle]['capacity']
                current_location = supplier['location']

    # Apply the soft cost limit penalty if the total cost exceeds the limit
    if total_cost > COST_LIMIT:
        cost_overrun = total_cost - COST_LIMIT
        total_cost += cost_overrun * cost_overrun_penalty(cost_overrun)

    fitness_score = (1 / (1 + total_cost)) + total_reward
    # Uncomment and define the `length_penalty` function if needed
    # long_penalty = length_penalty(decoded_length, encoded_length)
    # fitness_score -= long_penalty

    update_best_organism(encoded_genome, fitness_score, verbose=True)  # Update best organism, if applicable
    return fitness_score, {}



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
            vehicles=vehicles,  # Pass the vehicles here
            suppliers=suppliers,  # Pass the suppliers here
            seed=GLOBAL_SEED,
        )
        instructor_encodings = instructor_ga.instructor_phase()

        student_ga = ExperimentGA(
            genes=GENES,
            fitness_function=fitness_function,
            after_population_selection=capture_best_organism,
            mutation_prob=MUTATION_PROB,
            delimited_mutation_prob=DELIMITED_MUTATION_PROB,
            open_mutation_prob=0.000,
            capture_mutation_prob=0,
            delimiter_insert_prob=0,
            crossover_prob=.90,
            elitism_ratio=ELITISM_RATIO,
            base_gene_prob=BASE_GENE_PROB+.20,
            max_individual_length=MAX_INDIVIDUAL_LENGTH,
            population_size=POPULATION_SIZE,
            num_parents=NUM_PARENTS,
            max_generations=MAX_GENERATIONS,
            delimiters=False,
            delimiter_space=0,
            logging=LOGGING,
            generation_logging=GENERATION_LOGGING,
            mutation_logging=MUTATION_LOGGING,
            crossover_logging=CROSSOVER_LOGGING,
            individual_logging=INDIVIDUAL_LOGGING,
            experiment_name=f"{experiment_name}_Student_Cycle_{cycle}",
            vehicles=vehicles,  # Pass the vehicles here
            suppliers=suppliers,  # Pass the suppliers here
            seed=GLOBAL_SEED,
            )

        student_ga.student_phase(instructor_encodings)

        control_ga = ExperimentGA(
            genes=GENES,
            after_population_selection=capture_best_organism,
            fitness_function=fitness_function,
            mutation_prob=MUTATION_PROB,
            delimited_mutation_prob=DELIMITED_MUTATION_PROB,
            open_mutation_prob=0,
            capture_mutation_prob=0,
            delimiter_insert_prob=0,
            crossover_prob=CROSSOVER_PROB,
            elitism_ratio=ELITISM_RATIO,
            base_gene_prob=BASE_GENE_PROB ,
            max_individual_length=MAX_INDIVIDUAL_LENGTH,
            population_size=POPULATION_SIZE,
            num_parents=NUM_PARENTS,
            max_generations=MAX_GENERATIONS,
            delimiters=False,
            delimiter_space=0,
            logging=LOGGING,
            generation_logging=GENERATION_LOGGING,
            mutation_logging=MUTATION_LOGGING,
            crossover_logging=CROSSOVER_LOGGING,
            individual_logging=INDIVIDUAL_LOGGING,
            experiment_name=f"{experiment_name}_Control_Cycle_{cycle}",
            vehicles=vehicles,  # Pass the vehicles here
            suppliers=suppliers,  # Pass the suppliers here
            seed=GLOBAL_SEED,
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
    print(
        f"Results for Cycle {cycle}:\nInstructor Best Fitness: {max(instructor_ga.fitness_scores)}"
        f"\nStudent Best Fitness: {max(student_ga.fitness_scores)}\nControl Best Fitness: "
        f"{max(control_ga.fitness_scores)}")


def capture_best_organism(ga_instance):
    best_index = ga_instance.fitness_scores.index(max(ga_instance.fitness_scores))
    best_organism = ga_instance.population[best_index]

    # Decode the best organism using the instance's method
    decoded_organism = ga_instance.decode_organism(best_organism, format=False)



if __name__ == '__main__':
    # Initialize global variables for tasks, jobs, and machines
    # Initialize global variables for tasks, jobs, and machines
    vehicles = generate_vehicles(NUM_VEHICLES, (100, 200), (300, 500), (1.5, 3.0))
    suppliers = generate_suppliers(NUM_SUPPLIERS, (20, 50), ((8, 12), (14, 18)))
    GENES = list(vehicles.keys()) + list(suppliers.keys())
    experiment_name = input("Your_Experiment_Name: ")
    best_organisms = {}
    run_experiment(experiment_name, num_cycles, GENES, problem_specific_fitness_function)