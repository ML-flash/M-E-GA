
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:25:43 2024

@author: Matt Andrews
"""

import random
from M_E_GA_Base_V2 import M_E_GA_Base

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)  # Setting the seed for the main program

num_cycles=          2
MAX_GENERATIONS =    1000
LOGGING =            True
GENERATION_LOGGING = True
MUTATION_LOGGING =   False
CROSSOVER_LOGGING =  False
INDIVIDUAL_LOGGING = True 



# Assuming the M_E_GA_Base class definition is elsewhere and imported correctly

def generate_locations(num_locations, value_range=(10, 100), coord_range=(0, 100), fluctuation_range=20):
    locations = {}
    for i in range(num_locations):
        name = chr(65 + i)  # Generate location names A, B, C, etc.
        x, y = random.randint(*coord_range), random.randint(*coord_range)
        base_value_per_hour = random.randint(*value_range)
        fluctuation = random.randint(-fluctuation_range, fluctuation_range)
        locations[name] = {
            'coordinates': (x, y),
            'value_per_hour': base_value_per_hour,
            'fluctuation_range': fluctuation
        }
    return locations


def adjust_location_availability(locations, dropout_rate=0.1):
    # Randomly dropout a percentage of locations
    num_locations = len(locations)
    num_dropout = int(num_locations * dropout_rate)
    dropout_locations = random.sample(list(locations.keys()), num_dropout)
    
    for loc in dropout_locations:
        locations[loc]['available'] = False  # Mark as unavailable
    
    return locations



def get_fluctuating_value_per_hour(location, locations):
    base_value = locations[location]['value_per_hour']
    fluctuation = locations[location]['fluctuation_range']
    return base_value + random.uniform(-fluctuation, fluctuation)

def calculate_distance(loc1, loc2, locations):
    x1, y1 = locations[loc1]['coordinates']
    x2, y2 = locations[loc2]['coordinates']
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def calculate_travel(distance, speed):
    speeds = {'Slow': 1, 'Medium': 1.5, 'Fast': 3}
    travel_cost = distance * speeds[speed]
    travel_time = distance / speeds[speed]
    return travel_time, travel_cost

def is_move_valid(current_location, next_location, speed, locations, speeds):
    # Add a check to ensure next_location is not the same as current_location
    if next_location == current_location:
        return False  # This move is invalid as it attempts to stay in the same place
    return next_location in locations and speed in speeds

def perform_move(current_location, next_location, speed, locations):
    distance = calculate_distance(current_location, next_location, locations)
    travel_time, travel_cost = calculate_travel(distance, speed)
    return next_location, travel_time, travel_cost

def apply_costs_and_time(fitness_score, travel_cost, travel_time, hours_elapsed, day_counter, current_location, locations, verbosity=True):
    fitness_score -= travel_cost
    hours_elapsed += travel_time
    if hours_elapsed >= 24:
        day_counter += 1
        hours_elapsed %= 24
        if current_location:
            value_per_hour = get_fluctuating_value_per_hour(current_location, locations)
            income = value_per_hour * (24 - hours_elapsed)
            fitness_score += income
            if verbosity:
                print(f"Earned {income} points from staying at {current_location} for {24 - hours_elapsed} hours.")
    return fitness_score, hours_elapsed, day_counter

def handle_robbery(fitness_score, robbery_chance, current_location, robbery_fee, verbosity=True):
    if current_location and robbery_chance > random.random():
        fitness_score -= robbery_fee
        robbery_chance = 0.20
        if verbosity:
            print(f"Robbed at {current_location}! Lost {robbery_fee} points.")
    else:
        robbery_chance = min(robbery_chance + 0.30, 1.0)
    return fitness_score, robbery_chance

def problem_specific_fitness_function(encoded_individual, encoding_manager, total_days=90, verbosity=False, penalty=100):
    genome_length = len(encoded_individual)
    decoded_individual = encoding_manager.decode(encoded_individual)
    fitness_score = 1000
    current_location = 'A'
    hours_elapsed = 0
    day_counter = 1
    robbery_chance = 0.20  # Initial chance of being robbed
    robbery_fee = 5000
    just_moved = False  # Flag to indicate if the organism has just moved

    if verbosity:
        print("Starting simulation...")

    i = 0
    while day_counter <= total_days:
        if i < len(decoded_individual) - 1:
            location_gene = decoded_individual[i]
            speed_gene = decoded_individual[i + 1]

            # Check if the next location is available
            if not locations.get(location_gene, {}).get('available', True):
                # Apply a penalty for using an unavailable location
                fitness_score -= penalty
                if verbosity:
                    print(f"Penalty applied for attempting to use unavailable location: {location_gene}")

            if current_location != location_gene and is_move_valid(current_location, location_gene, speed_gene, locations, speeds):
                # Perform move
                previous_location = current_location
                next_location, travel_time, travel_cost = perform_move(current_location, location_gene, speed_gene, locations)
                current_location = next_location  # Update current_location after the move
                fitness_score, hours_elapsed, day_counter = apply_costs_and_time(fitness_score, travel_cost, travel_time, hours_elapsed, day_counter, None, locations, verbosity)
                
                if verbosity:
                    print(f"Day {day_counter}: Moved from {previous_location} to {next_location} at {speed_gene} speed. Travel Time: {travel_time} hours, Travel Cost: {travel_cost} points.")
                i += 2
                just_moved = True  # Set flag to True after moving
            else:
                # Stayed in place or invalid move
                if verbosity:
                    print(f"Day {day_counter}: Stayed in place at {current_location}.")
                i += 1
                just_moved = False  # Reset flag if staying in place or invalid move

        # Perform robbery check only if not just moved
        if not just_moved:
            fitness_score, robbery_chance = handle_robbery(fitness_score, robbery_chance, current_location, robbery_fee, verbosity)
            robbery_chance = min(robbery_chance + 0.10, 1.0)  # Increase robbery chance for the next day

        # Apply income for the time spent in the current location at the end of the day
        if hours_elapsed < 24:
            income = locations[current_location]['value_per_hour'] * (24 - hours_elapsed)
            fitness_score += income
            if verbosity:
                print(f"Day {day_counter}: Earned {income} points from staying at {current_location} for {24 - hours_elapsed} hours.")
            hours_elapsed = 24

        # Reset just_moved flag at the end of the day
        just_moved = False

        # Advance to the next day
        day_counter += 1
        hours_elapsed = 0
        fitness_score = fitness_score - (3 * genome_length)

    return fitness_score






# Generate locations
locations = generate_locations(500)

# Define genes (locations and speeds)
speeds = ['Slow', 'Medium', 'Fast']
genes = list(locations.keys()) + speeds

def run_experiment(experiment_name, num_cycles, genes, fitness_function, locations):
    for cycle in range(1, num_cycles + 1):
        print(f"\n--- Starting Experiment Cycle: {cycle} ---")

        # Instructor Phase
        print(f"\n--- Instructor Phase, Cycle: {cycle} ---")
        instructor_ga = M_E_GA_Base(
            genes=genes,
            fitness_function=fitness_function,
            mutation_prob=           0.015,
            delimited_mutation_prob= 0.014,
            open_mutation_prob=      0.007,
            capture_mutation_prob=   0.002,
            delimiter_insert_prob=   0.005,
            crossover_prob=          0.7,
            elitism_ratio=           0.05,
            base_gene_prob=          0.60,
            max_individual_length=   60,
            population_size=         600,
            num_parents=             200,
            max_generations = MAX_GENERATIONS,
            delimiters=False,
            delimiter_space=2,
            logging= LOGGING,
            generation_logging = GENERATION_LOGGING, 
            mutation_logging = MUTATION_LOGGING,
            crossover_logging = CROSSOVER_LOGGING, 
            individual_logging = INDIVIDUAL_LOGGING,
            experiment_name=f"{experiment_name}_Instructor_Cycle_{cycle}",
            seed = GLOBAL_SEED
        )
        instructor_ga.run_algorithm()
        instructor_encodings = instructor_ga.encoding_manager.encodings

        # Student Phase
        print(f"\n--- Student Phase, Cycle: {cycle} ---")
        student_ga = M_E_GA_Base(
            genes=genes,
            fitness_function=fitness_function,
            mutation_prob=           0.015,
            delimited_mutation_prob= 0.014,
            open_mutation_prob=      0.000,
            capture_mutation_prob=   0.00,
            delimiter_insert_prob=   0.00,
            crossover_prob=          0.7,
            elitism_ratio=           0.05,
            base_gene_prob=          0.50,
            max_individual_length=   60,
            population_size=         600,
            num_parents=             100,
            max_generations = MAX_GENERATIONS,
            delimiters= False,
            delimiter_space= 2,
            logging = LOGGING,
            generation_logging = GENERATION_LOGGING, 
            mutation_logging = MUTATION_LOGGING,
            crossover_logging = CROSSOVER_LOGGING, 
            individual_logging = INDIVIDUAL_LOGGING,
            experiment_name=f"{experiment_name}_Student_Cycle_{cycle}",
            encodings=instructor_encodings,
            seed = GLOBAL_SEED
        )
        student_ga.run_algorithm()

        # Control Phase
        print(f"\n--- Control Phase, Cycle: {cycle} ---")
        control_ga = M_E_GA_Base(
            genes=genes,
            fitness_function=fitness_function,
            mutation_prob=           0.015,
            delimited_mutation_prob= 0.015,
            open_mutation_prob=      0.000,
            capture_mutation_prob=   0.0,
            delimiter_insert_prob=   0.0,
            crossover_prob=          0.7,
            elitism_ratio=           0.05,
            base_gene_prob=          0.7,
            max_individual_length=   60,
            population_size=         400,
            num_parents=             100,
            max_generations = MAX_GENERATIONS,
            delimiters=False,
            delimiter_space=2,
            logging = LOGGING,
            generation_logging = GENERATION_LOGGING, 
            mutation_logging = MUTATION_LOGGING,
            crossover_logging = CROSSOVER_LOGGING, 
            individual_logging = INDIVIDUAL_LOGGING,
            experiment_name=f"{experiment_name}_Control_Cycle_{cycle}",
            seed = GLOBAL_SEED
            )
        control_ga.run_algorithm()

        # Log results and compare the phases
        print("\n--- Results Summary ---")
        compare_results(instructor_ga, student_ga, control_ga, cycle)

def compare_results(instructor_ga, student_ga, control_ga, cycle):
    # Implement your logic to compare and log results from each phase for the current cycle
    # Placeholder for demonstration purposes
    print(f"Results for Cycle {cycle}:\nInstructor Best Fitness: {max(instructor_ga.fitness_scores)}\nStudent Best Fitness: {max(student_ga.fitness_scores)}\nControl Best Fitness: {max(control_ga.fitness_scores)}")

if __name__ == '__main__':
    experiment_name = input("Your_Experiment_Name: ")
    num_cycles = num_cycles  # Adjust based on your requirement
    run_experiment(experiment_name, num_cycles, genes, problem_specific_fitness_function, locations)