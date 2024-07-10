import random
class TspAdvanced:
    def __init__(self, num_locations, value_range, coord_range, update_best_func):
        self.update_best = update_best_func
        self.locations = self.generate_locations(num_locations, value_range, coord_range)
        self.speeds = ['Slow', 'Medium', 'Fast']
        self.genes = list(self.locations.keys()) + self.speeds

    def generate_locations(self, num_locations, value_range=(10, 100), coord_range=(0, 100)):
        locations = {}
        for i in range(num_locations):
            name = chr(65 + i)  # Generate location names A, B, C, etc.
            x, y = random.randint(*coord_range), random.randint(*coord_range)
            value_per_hour = random.randint(*value_range)
            locations[name] = {'coordinates': (x, y), 'value_per_hour': value_per_hour}
        return locations

    def calculate_distance(self, loc1, loc2, locations):
        x1, y1 = locations[loc1]['coordinates']
        x2, y2 = locations[loc2]['coordinates']
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def calculate_travel(self, distance, speed):
        speeds = {'Slow': 1, 'Medium': 1.5, 'Fast': 3}
        travel_cost = distance * speeds[speed]
        travel_time = distance / speeds[speed]
        return travel_time, travel_cost

    def is_move_valid(self, current_location, next_location, speed, locations, speeds):
        # Add a check to ensure next_location is not the same as current_location
        if next_location == current_location:
            return False  # This move is invalid as it attempts to stay in the same place
        return next_location in locations and speed in speeds

    def perform_move(self, current_location, next_location, speed, locations):
        distance = self.calculate_distance(current_location, next_location, locations)
        travel_time, travel_cost = self.calculate_travel(distance, speed)
        return next_location, travel_time, travel_cost

    def apply_costs_and_time(self, fitness_score, travel_cost, travel_time, hours_elapsed, day_counter, current_location,
                             locations, verbosity=True):
        fitness_score -= travel_cost
        hours_elapsed += travel_time
        if hours_elapsed >= 24:
            day_counter += 1
            hours_elapsed %= 24
            if current_location:
                income = locations[current_location]['value_per_hour'] * (24 - hours_elapsed)
                fitness_score += income
                if verbosity:
                    print(f"Earned {income} points from staying at {current_location} for {24 - hours_elapsed} hours.")
        return fitness_score, hours_elapsed, day_counter

    def handle_robbery(self, fitness_score, robbery_chance, current_location, robbery_fee, verbosity=True):
        if current_location and robbery_chance > random.random():
            fitness_score -= robbery_fee
            robbery_chance = 0.20
            if verbosity:
                print(f"Robbed at {current_location}! Lost {robbery_fee} points.")
        else:
            robbery_chance = min(robbery_chance + 0.30, 1.0)
        return fitness_score, robbery_chance

    def compute(self, encoded_individual, encoding_manager, total_days=90, verbosity=False):
        decoded_individual = encoding_manager.decode_organism(encoded_individual)
        decoded_length = len(decoded_individual)
        fitness_score = 2000
        current_location = 'A'
        hours_elapsed = 0
        day_counter = 1
        robbery_chance = 0.30  # Initial chance of being robbed
        robbery_fee = 10000
        just_moved = False  # Flag to indicate if the organism has just moved

        if verbosity:
            print("Starting simulation...")

        i = 0
        while day_counter <= total_days:
            if i < len(decoded_individual) - 1:
                location_gene = decoded_individual[i]
                speed_gene = decoded_individual[i + 1]

                if current_location != location_gene and self.is_move_valid(current_location, location_gene, speed_gene,
                                                                            self.locations, self.speeds):
                    # Perform move
                    previous_location = current_location
                    next_location, travel_time, travel_cost = self.perform_move(current_location, location_gene,
                                                                                speed_gene,
                                                                                self.locations)
                    current_location = next_location  # Update current_location after the move
                    fitness_score, hours_elapsed, day_counter = self.apply_costs_and_time(fitness_score, travel_cost,
                                                                                          travel_time, hours_elapsed,
                                                                                          day_counter, None,
                                                                                          self.locations,
                                                                                          verbosity)

                    if verbosity:
                        # Use the updated current_location in the output message
                        print(
                            f"Day {day_counter}: Moved from {previous_location} to {next_location} at {speed_gene} speed. Travel Time: {travel_time} hours, Travel Cost: {travel_cost} points.")
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
                fitness_score, robbery_chance = self.handle_robbery(fitness_score, robbery_chance, current_location,
                                                                    robbery_fee, verbosity)
                robbery_chance = min(robbery_chance + 0.10, 1.0)  # Increase robbery chance for the next day

            # Apply income for the time spent in the current location at the end of the day
            if hours_elapsed < 24:
                income = self.locations[current_location]['value_per_hour'] * (24 - hours_elapsed)
                fitness_score += income
                if verbosity:
                    print(
                        f"Day {day_counter}: Earned {income} points from staying at {current_location} for {24 - hours_elapsed} hours.")
                hours_elapsed = 24

            # Reset just_moved flag at the end of the day
            just_moved = False

            # Advance to the next day
            day_counter += 1
            hours_elapsed = 0
            fitness_score -= decoded_length*.5
            self.update_best(encoded_individual, fitness_score, verbose=False)

        return fitness_score
