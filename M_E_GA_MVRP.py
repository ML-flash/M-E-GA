import random
from M_E_GA_Base_V2 import M_E_GA_Base
import math

GLOBAL_SEED =            None
num_cycles =              1
MAX_GENERATIONS =        1000
random.seed(GLOBAL_SEED)


MUTATION_PROB =           0.09
DELIMITED_MUTATION_PROB = 0.05
OPEN_MUTATION_PROB =      0.005
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


LOGGING =                 False
GENERATION_LOGGING =      True
MUTATION_LOGGING =        False
CROSSOVER_LOGGING =       False
INDIVIDUAL_LOGGING =      True


NUM_VEHICLES = 2000
NUM_SUPPLIERS = 6000
DEPOT_PENALTY = -100


best_organism = {
    "genome": None,
    "fitness": float('-inf')  # Start with negative infinity to ensure any valid organism will surpass it
}

distance_cache = {}


def length_penalty(decoded_length, encoded_length):
    return  -1.5 * (decoded_length - encoded_length)



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
                                      depot_return_reward=.1, distance_penalty_factor=0.01, DEPOT_PENALTY=-10):
    decoded_genome = ga_instance.decode_organism(encoded_genome)
    encoded_length = len(encoded_genome)
    decoded_length = len(decoded_genome)
    total_cost = 0
    total_reward = 0
    current_location = depot_location
    vehicle_distance = {}
    vehicle_load = {}
    visited_suppliers = set()

    last_vehicle = None
    returned_to_depot = False
    previous_gene_was_s0 = False  # Track consecutive S0s

    for i, gene in enumerate(decoded_genome):
        if gene in ['Start', 'End']:
            decoded_length -= 1
            encoded_length -= 1
            continue

        if gene == "S0":
            if previous_gene_was_s0:  # Apply consecutive S0 penalty
                total_cost += DEPOT_PENALTY
            previous_gene_was_s0 = True

            if last_vehicle:
                distance_to_depot = calculate_distance_cached(current_location, depot_location)
                total_cost += distance_to_depot * vehicles[last_vehicle]['cost_per_distance']
                if vehicle_distance[last_vehicle] <= vehicles[last_vehicle]['max_distance']:
                    total_reward += depot_return_reward * vehicle_load[last_vehicle]  # Reward based on load
                vehicle_load[last_vehicle] = 0
                vehicle_distance[last_vehicle] = 0  # Reset vehicle distance
                returned_to_depot = True
            current_location = depot_location

        elif gene in vehicles:
            previous_gene_was_s0 = False
            if last_vehicle and not returned_to_depot:
                distance_to_depot = calculate_distance_cached(current_location, depot_location)
                total_cost += distance_to_depot * vehicles[last_vehicle]['cost_per_distance']
                if vehicle_distance[last_vehicle] <= vehicles[last_vehicle]['max_distance']:
                    total_reward += (depot_return_reward * vehicle_load[last_vehicle]) / 2  # Half reward for load
                    total_reward -= distance_penalty_factor * distance_to_depot  # Penalty for distance
                vehicle_load[last_vehicle] = 0
                vehicle_distance[last_vehicle] = 0  # Reset vehicle distance

            last_vehicle = gene
            returned_to_depot = False
            current_location = depot_location
            vehicle_distance[last_vehicle] = 0  # Initialize for new vehicle
            vehicle_load[last_vehicle] = 0  # Initialize for new vehicle

        elif gene in suppliers and last_vehicle:
            previous_gene_was_s0 = False
            supplier = suppliers[gene]
            distance = calculate_distance_cached(current_location, supplier['location'])
            if vehicle_distance[last_vehicle] + distance <= vehicles[last_vehicle]['max_distance']:  # Check distance limit
                vehicle_distance[last_vehicle] += distance
                if gene not in visited_suppliers:
                    visited_suppliers.add(gene)
                    vehicle_load[last_vehicle] += supplier['demand']
                    if vehicle_load[last_vehicle] > vehicles[last_vehicle]['capacity']:
                        total_cost += penalty_per_unit_demand_excess * (vehicle_load[last_vehicle] - vehicles[last_vehicle]['capacity'])
                        vehicle_load[last_vehicle] = vehicles[last_vehicle]['capacity']  # Cap at max capacity
                total_cost += distance * vehicles[last_vehicle]['cost_per_distance']
                current_location = supplier['location']
            else:
                vehicle_load[last_vehicle] = 0  # Exceeded max distance, load set to 0

    # Fitness score calculation with penalties
    fitness_score = (1 / (1 + total_cost)) + total_reward
    penalty = length_penalty(decoded_length, encoded_length)  # Apply length penalty
    fitness_score -= penalty  # Reduce fitness score by penalty

    update_best_organism(encoded_genome, fitness_score, verbose=True)
    return fitness_score, {}


# Initialize global variables for tasks, jobs, and machines
vehicles = generate_vehicles(NUM_VEHICLES, (100, 200), (300, 500), (1.5, 3.0))
suppliers = generate_suppliers(NUM_SUPPLIERS, (20, 50), ((8, 12), (14, 18)))
GENES = list(vehicles.keys()) + list(suppliers.keys())


# GA configuration
config = {
    'genes': GENES,
    'fitness_function': lambda ind, ga: problem_specific_fitness_function(ind, ga, vehicles, suppliers,
                        depot_location=(0, 0), penalty_per_unit_time=10, penalty_per_unit_demand_excess=20,
                        resupply_cost=50, base_reset_cost=5, cost_per_unit_distance=0.1, service_time=1),
    'mutation_prob': MUTATION_PROB,
    'delimited_mutation_prob': DELIMITED_MUTATION_PROB,
    'open_mutation_prob': OPEN_MUTATION_PROB,
    'capture_mutation_prob': CAPTURE_MUTATION_PROB,
    'delimiter_insert_prob': DELIMITER_INSERT_PROB,
    'crossover_prob': CROSSOVER_PROB,
    'elitism_ratio': ELITISM_RATIO,
    'base_gene_prob': BASE_GENE_PROB,
    'max_individual_length': MAX_INDIVIDUAL_LENGTH,
    'population_size': POPULATION_SIZE,
    'num_parents': NUM_PARENTS,
    'max_generations': MAX_GENERATIONS,
    'delimiters': DELIMITERS,
    'delimiter_space': DELIMITER_SPACE,
    'logging': LOGGING,
    'generation_logging': GENERATION_LOGGING,
    'mutation_logging': MUTATION_LOGGING,
    'crossover_logging': CROSSOVER_LOGGING,
    'individual_logging': INDIVIDUAL_LOGGING,
    'seed': GLOBAL_SEED
}

# Initialize GA
ga = M_E_GA_Base(**config)

# Execute the GA
ga.run_algorithm()

# Analyze the results
best_genome = best_organism["genome"]
best_fitness = best_organism["fitness"]
best_solution_decoded = ga.decode_organism(best_genome, format=True)
print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
verbose_fitness_score, _ = problem_specific_fitness_function(best_genome, ga, vehicles, suppliers,
                        depot_location=(0, 0), penalty_per_unit_time=10, penalty_per_unit_demand_excess=20,
                        resupply_cost=50, base_reset_cost=5, cost_per_unit_distance=0.1, service_time=1)
print(f"Verbose Fitness Evaluation Score: {verbose_fitness_score}")
print(f"Best Genome (Encoded): {best_genome}")
