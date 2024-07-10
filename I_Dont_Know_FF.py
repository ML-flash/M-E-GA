import random
class IDontKnow:
    """Alternative fitness function class for testing a different genetic representation or computation strategy."""
    def __init__(self, volume, num_items, num_groups, update_best_func):
        self.update_best = update_best_func
        self.volume = volume
        self.num_items = num_items
        self.num_groups = num_groups

        self.genes = ['R', 'L', 'U', 'D', 'F', 'B']
        self.directions = {'R': (1, 0, 0), 'L': (-1, 0, 0), 'U': (0, 1, 0), 'D': (0, -1, 0), 'F': (0, 0, 1), 'B': (0, 0, -1)}

        self.items = self.create_items()
        for item in self.items:
            item['position'] = (random.randint(-self.volume, self.volume),
                                random.randint(self.volume, self.volume),
                                random.randint(-self.volume, self.volume))

        all_positions = [(x, y, z) for x in range(-volume, volume + 1)
                         for y in range(-volume, volume + 1)
                         for z in range(-volume, volume + 1)]

        # Randomly shuffle the positions
        random.shuffle(all_positions)

        # Ensure that the number of items does not exceed the number of available positions
        if num_items > len(all_positions):
            raise ValueError("Number of items exceeds the number of available positions in the given volume.")

        # Assign each item a unique position from the shuffled list
        self.items = self.create_items()
        for item, position in zip(self.items, all_positions[:num_items]):
            item['position'] = position

    def create_items(self, properties=('size', 'weight', 'density', 'value')):
        items = []
        for item_id in range(self.num_items):
            item = {
                'id': item_id,
                'group': random.randint(0, self.num_groups - 1),
                'properties': {
                    'size': random.uniform(0.1, 10.0),
                    'weight': random.uniform(0.1, 30.0),
                    'density': random.uniform(0.1, 10.0),
                    'value': random.uniform(0.1, 80.0),
                },
                'reaction_strength': random.uniform(0.5, 5.5),
                'interactions': []
            }
            num_interactions = random.randint(0, self.num_groups - 1)
            interacting_groups = random.sample([group for group in range(self.num_groups) if group != item['group']], num_interactions)
            for target_group in interacting_groups:
                interaction = {
                    'target_group': target_group,
                    'property': random.choice(properties),
                    'direction': random.choice(['increase', 'decrease']),
                    'magnitude': random.uniform(0.1, 0.5),
                }
                item['interactions'].append(interaction)
            items.append(item)
        return items

    def can_add_item_to_sack(self, item, sack):
        # Example based on item size and sack's remaining capacity
        return (sack['current_capacity'] + item['properties']['size']) <= sack['max_capacity']

    def collect_item(self, item, sack, verbose=False):
        sack['items'].append(item)  # Add item to sack
        sack['current_capacity'] += item['properties']['size']  # Update sack's capacity usage
        if verbose:
            print(
                f"Collected item {item['id']} with size {item['properties']['size']}. Current sack capacity: {sack['current_capacity']}/{sack['max_capacity']}")
        # Optionally, apply item's interactions immediately upon collection
        self.apply_interactions(item, sack['items'], verbose=verbose)

    def apply_interactions(self, new_item, items_in_sack, verbose=False):
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
                        print(
                            f"Applying interaction from item {new_item['id']} to item {item['id']}: {affected_property} {'increased' if change > 0 else 'decreased'} from {old_value} to {item['properties'][affected_property]}")

    def calculate_sack_value(self,sack, verbose=False):
        total_value = sum(item['properties']['value'] for item in sack['items'])
        # Apply penalties or adjustments based on sack constraints if necessary
        if verbose:
            print(f"Total sack value: {total_value}")
        return total_value

    def compute(self, encoded_individual, ga_instance, sack_capacity=152, verbose=False):
        decoded_individual = ga_instance.decode_organism(encoded_individual)
        fitness_score = 0
        volume_bound = self.volume
        x, y, z = 0, 0, 0  # Starting position
        visited_positions = set([(x, y, z)])  # Track visited positions to avoid self-collision
        sack = {'items': [], 'current_capacity': 0, 'max_capacity': sack_capacity}  # Initialize the sack

        for gene in decoded_individual:
            if gene in self.directions:
                dx, dy, dz = self.directions[gene]  # Get the direction vector
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
                for item in self.items:
                    if 'position' in item and item['position'] == new_pos and self.can_add_item_to_sack(item, sack):
                        self.collect_item(item, sack, verbose=verbose)  # Collect the item if it fits in the sack

                fitness_score += .5  # Reward for each valid move
                visited_positions.add(new_pos)  # Mark the new position as visited
                x, y, z = new_pos  # Update the current position

        fitness_score += self.calculate_sack_value(sack, verbose=verbose)  # Add value of collected items to the fitness score

        # Verbose logging for the final fitness score
        if verbose:
            print(f"Final fitness score: {fitness_score}")
        self.update_best(encoded_individual, fitness_score, verbose=False)

        return fitness_score


