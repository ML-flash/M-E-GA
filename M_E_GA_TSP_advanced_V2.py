# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:25:43 2024

@author: Matt Andrews
"""

import random
from M_E_GA_Base_V2 import M_E_GA_Base

# Assuming the M_E_GA_Base class definition is elsewhere and imported correctly

def generate_locations(num_locations, value_range=(10, 100), coord_range=(0, 1000), fluctuation_range=30):
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
locations = generate_locations(1000)

# Define genes (locations and speeds)
speeds = ['Slow', 'Medium', 'Fast']
genes = list(locations.keys()) + speeds

# Standard configuration settings
config = {
    'mutation_prob':           0.015,
    'delimited_mutation_prob': 0.014,
    'open_mutation_prob':      0.07,
    'capture_mutation_prob':   0.05,
    'delimiter_insert_prob':   0.04,
    'crossover_prob':          0.7,
    'elitism_ratio':           0.05,
    'base_gene_prob':          0.55,
    'max_individual_length':   60,
    'population_size':         600,
    'num_parents':             100,
    'max_generations':         500,
    'delimiters':              False,
    'delimiter_space':         2,
}

# Initialize and run the GA
ga = M_E_GA_Base(
    genes=genes,
    fitness_function=problem_specific_fitness_function,
    **config
)

ga.run_algorithm()

# Result Analysis
best_fitness = max(ga.fitness_scores)
best_index = ga.fitness_scores.index(best_fitness)
final_fitness_score = problem_specific_fitness_function(ga.population[best_index], ga.encoding_manager, total_days=90, verbosity=True)

print(f"Final Fitness Score: {final_fitness_score}")
best_solution = ga.decode_organism(ga.population[best_index], format=False)
print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")
