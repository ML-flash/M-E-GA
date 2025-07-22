import random
import json
import os
import datetime
import uuid

class IDontKnow:
    def __init__(
        self, volume, num_items, num_groups,
        update_best_func, max_size, max_weight, max_density,
        enable_logging=True, fitness_only_logging=True
    ):
        self.update_best = update_best_func
        self.volume = volume
        self.num_items = num_items
        self.num_groups = num_groups
        self.max_size = max_size
        self.max_weight = max_weight
        self.max_density = max_density
        self.enable_logging = enable_logging
        self.fitness_only_logging = fitness_only_logging
        
        # Persistent position tracking across evaluations
        self.current_position = (0, 0, 0)

        self.genes = ['R', 'L', 'U', 'D', 'F', 'B', 'DR']
        self.directions = {
            'R': (1, 0, 0), 'L': (-1, 0, 0),
            'U': (0, 1, 0), 'D': (0, -1, 0),
            'F': (0, 0, 1), 'B': (0, 0, -1)
        }

        # Initialize items and positions
        self.items = self.create_items()
        all_positions = [
            (x, y, z)
            for x in range(-self.volume, self.volume + 1)
            for y in range(-self.volume, self.volume + 1)
            for z in range(-self.volume, self.volume + 1)
        ]
        random.shuffle(all_positions)
        if self.num_items > len(all_positions):
            raise ValueError("Number of items exceeds available positions.")
        for item, pos in zip(self.items, all_positions[:self.num_items]):
            item['position'] = pos

        # Set up logging filenames
        if self.enable_logging:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            if self.fitness_only_logging:
                self.log_filename = f"logs/fitness_log_{ts}.jsonl"
            else:
                self.log_filename = f"logs/evaluation_log_{ts}.jsonl"
                self.step_log_filename = f"logs/step_log_{ts}.jsonl"
            self.ensure_log_directory()

        # Rewards & penalties
        self.step_reward = 2
        self.step_penalty = -4
        self.soft_step_limit = 200
        self.outside_step_penalty = -4
        self.penalty_per_remaining_gene = -1

    def ensure_log_directory(self):
        if not self.enable_logging:
            return
        files = [self.log_filename]
        if hasattr(self, 'step_log_filename'):
            files.append(self.step_log_filename)
        for fn in files:
            d = os.path.dirname(fn)
            if not os.path.exists(d):
                os.makedirs(d)

    def log_step(self, step_info):
        # Per-step detailed logging (skipped if fitness-only)
        if not self.enable_logging or self.fitness_only_logging:
            return
        with open(self.step_log_filename, 'a') as f:
            f.write(json.dumps(step_info) + '\n')

    def wrap_coordinate(self, coord):
        # Robust toroidal wrapping using modulo
        size = 2 * self.volume + 1
        return ((coord + self.volume) % size) - self.volume

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
            groups = [g for g in range(self.num_groups) if g != item['group']]
            for tgt in random.sample(groups, num_interactions):
                item['interactions'].append({
                    'target_group': tgt,
                    'property': random.choice(properties),
                    'direction': random.choice(['increase', 'decrease']),
                    'magnitude': random.uniform(0.01, 10.0)
                })
            items.append(item)
        return items

    def can_add_item_to_sack(self, item, sack):
        if (sack['current_size'] + item['properties']['size'] > sack['max_size'] or
            sack['current_weight'] + item['properties']['weight'] > sack['max_weight'] or
            sack['current_density'] + item['properties']['density'] > sack['max_density']):
            return False
        return True

    def collect_item(self, item, sack, verbose=False):
        sack['items'].append(item)
        sack['current_size'] += item['properties']['size']
        sack['current_weight'] += item['properties']['weight']
        sack['current_density'] += item['properties']['density']

    def _apply_single_interaction(self, source_item, target_item, interaction):
        prop = interaction['property']
        mag = interaction['magnitude'] * source_item['reaction_strength']
        change = mag if interaction['direction']=='increase' else -mag
        old = target_item['properties'][prop]
        target_item['properties'][prop] = max(0, old + change)

    def calculate_sack_value(self, sack, verbose=False):
        if not sack['items']:
            return 0
        self.apply_final_interactions(sack['items'])
        return sum(i['properties']['value'] for i in sack['items'])

    def apply_final_interactions(self, items):
        if len(items) <= 1:
            return
        for _ in range(5):
            any_change = False
            for i, src in enumerate(items):
                for j, tgt in enumerate(items):
                    if i==j: continue
                    for inter in src['interactions']:
                        if tgt['group']== inter['target_group']:
                            old = tgt['properties'][inter['property']]
                            self._apply_single_interaction(src, tgt, inter)
                            new = tgt['properties'][inter['property']]
                            if abs(new-old)>0.01:
                                any_change = True
            if not any_change:
                break

    def drop_oldest_item(self, sack, pos, items, verbose=False):
        if not sack['items']:
            return False
        oldest = sack['items'][0]
        # ensure no collision
        if any(it['position']==pos and it['id']!=oldest['id'] for it in items):
            return False
        sack['items'].pop(0)
        sack['current_size'] -= oldest['properties']['size']
        sack['current_weight'] -= oldest['properties']['weight']
        sack['current_density'] -= oldest['properties']['density']
        oldest['position'] = pos
        return True

    def compute(
        self, encoded_individual, ga_instance,
        sack_capacity=152, max_weight=200, max_density=50,
        verbose=False
    ):
        decoded = ga_instance.decode_organism(encoded_individual)
        fitness = 0.0
        eval_id = str(uuid.uuid4())

        x,y,z = self.current_position
        start_pos = (x,y,z)
        sack = {
            'items': [], 'current_size':0, 'max_size':sack_capacity,
            'current_weight':0, 'max_weight':max_weight,
            'current_density':0, 'max_density':max_density
        }
        drop_reward = 0.001
        dropped = set()
        collected = set()
        original_positions = {it['id']: it['position'] for it in self.items}

        step_count = 0

        for gene in decoded:
            before = (x,y,z)
            action = None
            item_id = None

            if gene=='DR' and sack['items']:
                itm = sack['items'][0]
                if self.drop_oldest_item(sack, (x,y,z), self.items, verbose):
                    fitness += drop_reward
                    action = 'drop'
                    item_id = itm['id']
                    dropped.add(itm['id'])
                    collected.discard(itm['id'])

            elif gene in self.directions:
                dx,dy,dz = self.directions[gene]
                x += dx; y += dy; z += dz
                x = self.wrap_coordinate(x)
                y = self.wrap_coordinate(y)
                z = self.wrap_coordinate(z)
                action = 'move'
                if step_count < self.soft_step_limit:
                    fitness += self.step_reward
                else:
                    fitness += self.step_penalty

                # collect logic
                pos = (x,y,z)
                itm = next((it for it in self.items if it['position']==pos), None)
                if itm and itm['id'] not in collected and self.can_add_item_to_sack(itm, sack):
                    self.collect_item(itm, sack, verbose)
                    collected.add(itm['id'])
                    action = 'collect'
                    item_id = itm['id']

            step_count += 1
            after = (x,y,z)
            # detailed per-step log
            self.log_step({
                'evaluation_id': eval_id,
                'step': step_count,
                'gene': gene,
                'position_before': before,
                'position_after': after,
                'action': action,
                'item_id': item_id,
                'sack_count': len(sack['items'])
            })

        # final evaluation and interactions
        if step_count>0:
            base_val = self.calculate_sack_value(sack, verbose)
            fitness += base_val
            if sack['items']:
                fitness *= (1 + len(sack['items'])*0.1)

        self.current_position = (x,y,z)
        # reset positions
        for it in self.items:
            if it['id'] not in dropped:
                it['position'] = original_positions[it['id']]

        # final summary log
        if self.enable_logging:
            self.log_evaluation(
                eval_id, decoded, sack, fitness,
                start_pos, (x,y,z), original_positions,
                dropped, step_count
            )

        self.update_best(encoded_individual, fitness)
        return fitness

    def log_evaluation(
        self, eval_id, decoded_individual, sack, fitness,
        start_pos, end_pos, original_positions,
        dropped_items, step_count
    ):
        if not self.enable_logging:
            return
        if self.fitness_only_logging:
            entry = {
                'evaluation_id': eval_id,
                'fitness_score': fitness,
                'timestamp': datetime.datetime.now().isoformat(),
                'num_items_collected': len(sack['items']),
                'step_count': step_count,
                'start_position': start_pos,
                'final_position': end_pos
            }
        else:
            entry = {
                'evaluation_id': eval_id,
                'path': decoded_individual,
                'start_position': start_pos,
                'final_position': end_pos,
                'items_start': [
                    {'id': it['id'], 'position': original_positions[it['id']]} for it in self.items
                ],
                'items_end': [
                    {'id': it['id'], 'position': it['position']} for it in self.items
                ],
                'sack_summary': {
                    'num_items': len(sack['items']),
                    'items': [it['id'] for it in sack['items']],
                    'total_value': sum(it['properties']['value'] for it in sack['items']),
                    'total_size': sack['current_size'],
                    'total_weight': sack['current_weight'],
                    'total_density': sack['current_density']
                },
                'dropped_items': list(dropped_items),
                'max_size': sack['max_size'],
                'max_weight': sack['max_weight'],
                'max_density': sack['max_density'],
                'fitness_score': fitness,
                'step_count': step_count,
                'volume_type': 'toroidal'
            }
        with open(self.log_filename, 'a') as f:
            f.write(json.dumps(entry) + '\n')
