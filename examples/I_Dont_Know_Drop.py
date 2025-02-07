import random
import json
import os
import datetime
import uuid

class IDontKnow:
    def __init__(self, volume, num_items, num_groups, update_best_func, max_size, max_weight, max_density):
        self.update_best = update_best_func
        self.volume = volume
        self.num_items = num_items
        self.num_groups = num_groups
        self.max_size = max_size
        self.max_weight = max_weight
        self.max_density = max_density

        self.genes = ['R', 'L', 'U', 'D', 'F', 'B', 'DR']
        self.directions = {'R': (1, 0, 0), 'L': (-1, 0, 0), 'U': (0, 1, 0), 'D': (0, -1, 0),
                           'F': (0, 0, 1), 'B': (0, 0, -1)}

        self.items = self.create_items()
        all_positions = [(x, y, z) for x in range(-self.volume, self.volume + 1)
                         for y in range(-self.volume, self.volume + 1)
                         for z in range(-self.volume, self.volume + 1)]
        random.shuffle(all_positions)
        if self.num_items > len(all_positions):
            raise ValueError("Number of items exceeds the number of available positions in the given volume.")
        for item, position in zip(self.items, all_positions[:self.num_items]):
            item['position'] = position

        self.log_filename = f"logs/evaluation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.ensure_log_directory()

        # Step rewards/penalties
        self.step_reward = 2          # Reward for each step up to the soft limit
        self.step_penalty = -4        # Penalty for each step beyond the soft limit
        self.soft_step_limit = 300    # Soft limit for steps, adjust as needed
        self.outside_step_penalty = -4  # Additional penalty for each step outside the box
        self.penalty_per_remaining_gene = -1  # Penalty for each remaining gene after stopping at boundary

    def ensure_log_directory(self):
        log_dir = os.path.dirname(self.log_filename)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def create_items(self, properties=('size', 'weight', 'density', 'value')):
        items = []
        for item_id in range(self.num_items):
            item = {
                'id': item_id,
                'group': random.randint(0, self.num_groups - 1),
                'properties': {
                    'size': random.uniform(1, 30),
                    'weight': random.uniform(1, 47),
                    'density': random.uniform(1, 50),
                    'value': random.uniform(0.1, 2000.0),
                },
                'reaction_strength': random.uniform(0.01, 50.0),
                'interactions': []
            }
            num_interactions = random.randint(0, self.num_groups - 1)
            interacting_groups = random.sample(
                [group for group in range(self.num_groups) if group != item['group']],
                num_interactions
            )
            for target_group in interacting_groups:
                interaction = {
                    'target_group': target_group,
                    'property': random.choice(properties),
                    'direction': random.choice(['increase', 'decrease']),
                    'magnitude': random.uniform(0.01, 10.0),
                }
                item['interactions'].append(interaction)
            items.append(item)
        return items

    def can_add_item_to_sack(self, item, sack):
        if (sack['current_size'] + item['properties']['size']) > sack['max_size'] or \
           (sack['current_weight'] + item['properties']['weight']) > sack['max_weight'] or \
           (sack['current_density'] + item['properties']['density']) > sack['max_density']:
            return False
        return True

    def collect_item(self, item, sack, verbose=False):
        sack['items'].append(item)
        sack['current_size'] += item['properties']['size']
        sack['current_weight'] += item['properties']['weight']
        sack['current_density'] += item['properties']['density']
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

    def calculate_sack_value(self, sack, verbose=False):
        total_value = sum(item['properties']['value'] for item in sack['items'])
        return total_value

    def drop_oldest_item(self, sack, current_position, items, verbose=False):
        if not sack['items']:
            return False

        item_at_position = next((item for item in items if item['position'] == current_position), None)
        if item_at_position:
            return False

        oldest_item = sack['items'].pop(0)
        sack['current_size'] -= oldest_item['properties']['size']
        sack['current_weight'] -= oldest_item['properties']['weight']
        sack['current_density'] -= oldest_item['properties']['density']

        oldest_item['position'] = current_position

        return True

    def compute(self, encoded_individual, ga_instance, sack_capacity=152, max_weight=200, max_density=50, verbose=False):
        decoded_individual = ga_instance.decode_organism(encoded_individual)
        fitness_score = 0.00
        x, y, z = 0, 0, 0
        visited_positions = set()
        sack = {
            'items': [],
            'current_size': 0,
            'max_size': sack_capacity,
            'current_weight': 0,
            'max_weight': max_weight,
            'current_density': 0,
            'max_density': max_density
        }
        drop_reward = 0.001
        final_position = (x, y, z)
        step_count = 0

        # Store original positions of all items
        original_positions = {item['id']: item['position'] for item in self.items}
        dropped_items = set()

        genes_processed = 0
        total_genes = len(decoded_individual)

        def wrap_coordinate(coord):
            if coord > self.volume:
                return -self.volume
            elif coord < -self.volume:
                return self.volume
            else:
                return coord

        for gene in decoded_individual:
            if gene == 'DR':
                if sack['items']:  # Check if there are items in the sack to drop
                    item_to_drop = sack['items'][0]  # Get the oldest item (first in the list)
                    if self.drop_oldest_item(sack, (x, y, z), self.items, verbose):
                        fitness_score += drop_reward
                        dropped_items.add(item_to_drop['id'])  # Add the ID of the dropped item
            elif gene in self.directions:
                dx, dy, dz = self.directions[gene]
                x += dx
                y += dy
                z += dz

                # Wrap coordinates to make the volume toroidal
                x = wrap_coordinate(x)
                y = wrap_coordinate(y)
                z = wrap_coordinate(z)
                new_pos = (x, y, z)

                # Apply step reward/penalty based on step count
                if step_count < self.soft_step_limit:
                    fitness_score += self.step_reward
                else:
                    fitness_score += self.step_penalty

                # Handle item collection
                item_at_position = next((item for item in self.items if item['position'] == new_pos), None)
                if item_at_position:
                    if self.can_add_item_to_sack(item_at_position, sack):
                        self.collect_item(item_at_position, sack, verbose)

                visited_positions.add(new_pos)
                final_position = (x, y, z)
                step_count += 1
            genes_processed += 1

        if step_count > 0:
            fitness_score += self.calculate_sack_value(sack, verbose)
            fitness_score *= 1.75 ** len(sack['items'])

        # Reset positions of collected items, keep dropped items in their new positions
        for item in self.items:
            if item['id'] not in dropped_items:
                item['position'] = original_positions[item['id']]

        self.log_evaluation(decoded_individual, sack, fitness_score, final_position, original_positions, dropped_items, step_count)

        self.update_best(encoded_individual, fitness_score)

        return fitness_score

    def log_evaluation(self, decoded_individual, sack, fitness_score, final_position, original_positions, dropped_items, step_count):
        log_entry = {
            "evaluation_id": str(uuid.uuid4()),
            "path": decoded_individual,
            "items_start": [{"id": item['id'], "position": original_positions[item['id']]} for item in self.items],
            "items_end": [{"id": item['id'], "position": item['position']} for item in self.items],
            "sack_summary": {
                "num_items": len(sack['items']),
                "items": [item['id'] for item in sack['items']],
                "total_value": sum(item['properties']['value'] for item in sack['items']),
                "total_size": sack['current_size'],
                "total_weight": sack['current_weight'],
                "total_density": sack['current_density']
            },
            "dropped_items": list(dropped_items),
            "max_size": sack['max_size'],
            "max_weight": sack['max_weight'],
            "max_density": sack['max_density'],
            "fitness_score": fitness_score,
            "final_position": final_position,
            "step_count": step_count,
            "volume_type": "toroidal"
        }

        with open(self.log_filename, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
