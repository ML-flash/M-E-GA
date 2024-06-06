# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:09:43 2024

@author: Matt Andrews
"""

# GNU GENERAL PUBLIC LICENSE
# By running this code, you acknowledge and agree to the terms of the LICENSE file
# provided in the repository. 


import json
import datetime
import random
import os
import concurrent.futures
from M_E_engine_V2 import EncodingManager


class M_E_GA_Base:
    def __init__(self, genes, fitness_function, mutation_prob=0.01, delimited_mutation_prob=0.01,
                 open_mutation_prob=0.0001, capture_mutation_prob=0.00001,
                 delimiter_insert_prob=0.00001, crossover_prob=0.50,
                 elitism_ratio=0.06, base_gene_prob=0.98,
                 max_individual_length=6, population_size=400,
                 num_parents=80, max_generations=1000,
                 delimiters=True, delimiter_space=3, logging=True,
                 generation_logging=True, mutation_logging=False,
                 crossover_logging=False, individual_logging=False,
                 experiment_name=None, encodings=None, seed=None,
                 before_fitness_evaluation=None, after_population_selection=None,
                 before_generation_finalize=None, capture_gene_prob=0, **kwargs):  # Added capture_gene_prob here
        # Directly use the provided genes list for the encoding manager without assuming a specific structure like 'gene['id']'
        self.genes = genes
        self.fitness_function = fitness_function
        self.logging = logging
        self.logs = []
        self.encoding_manager = EncodingManager()
        self.log_filename = ""
        self.experiment_name = experiment_name
        self.before_fitness_evaluation = before_fitness_evaluation
        self.after_population_selection = after_population_selection
        self.before_generation_finalize = before_generation_finalize

        # Set configuration parameters
        self.mutation_prob = mutation_prob
        self.delimited_mutation_prob = delimited_mutation_prob
        self.open_mutation_prob = open_mutation_prob
        self.capture_mutation_prob = capture_mutation_prob
        self.delimiter_insert_prob = delimiter_insert_prob
        self.crossover_prob = crossover_prob
        self.elitism_ratio = elitism_ratio
        self.base_gene_prob = base_gene_prob
        self.max_individual_length = max_individual_length
        self.population_size = population_size
        self.num_parents = num_parents
        self.max_generations = max_generations
        self.delimiters = delimiters
        self.delimiter_space = delimiter_space
        self.population = []
        self.current_generation = 0
        self.generation_logging = generation_logging
        self.mutation_logging = mutation_logging
        self.crossover_logging = crossover_logging
        self.individual_logging = individual_logging
        self.seed = seed
        self.relevant_data = None
        self.capture_gene_prob = capture_gene_prob

        # Seed used for reproducibility.
        if seed is not None:
            random.seed(seed)

        # Integrate encodings if provided, and add genes to the encoding manager
        if encodings:
            self.encoding_manager.integrate_uploaded_encodings(encodings, self.genes)
        else:
            for gene in self.genes:
                self.encoding_manager.add_gene(gene, verbose=True)

        if self.logging and self.experiment_name == None:
            self.experiment_name = input("Enter the experiment name: ")
            self.log_filename = f"{self.experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

    # Logging

    def log_generation(self, generation, fitness_scores, population=None):
        if self.logging and self.generation_logging:
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            median_fitness = sorted(fitness_scores)[len(fitness_scores) // 2]
            best_fitness = max(fitness_scores)
            worst_fitness = min(fitness_scores)

            summary_log = {
                "average_fitness": average_fitness,
                "median_fitness": median_fitness,
                "best_fitness": best_fitness,
                "worst_fitness": worst_fitness,
            }

            current_generation_log = self.logs[-1]  # Assuming this is called at the end of each generation
            current_generation_log["summary"] = summary_log

    def log_mutation(self, mutation_details):
        if self.logging and self.mutation_logging:
            # Find the latest generation log
            if self.logs:
                current_generation_log = self.logs[-1]
                current_generation_log["mutations"].append(mutation_details)

    def log_crossover(self, generation, parent1, parent2, crossover_point, offspring1, offspring2):
        if self.logging and self.crossover_logging:
            crossover_log = {
                "crossover_point": crossover_point,
                "parent1_before": parent1[:crossover_point] if crossover_point is not None else parent1,
                "parent2_before": parent2[:crossover_point] if crossover_point is not None else parent2,
                "offspring1": offspring1,
                "offspring2": offspring2,
            }
            current_generation_log = self.logs[-1]
            current_generation_log["crossovers"].append(crossover_log)

    def log_fitness_function_settings(self, settings):
        if self.logging and self.fitness_settings_logging and not self.fitness_settings_logged:
            # Include new parameters in the settings log
            settings.update({
                "MAX_VOLUME": self.max_volume,
                "VOLUME_PENALTY_FACTOR": self.volume_penalty_factor,
                "MAX_SIZE": self.max_size,
                "SIZE_PENALTY_FACTOR": self.size_penalty_factor
            })
            self.logs.append({"fitness_function_settings": settings})
            self.fitness_settings_logged = True

    def log_final_organism(self, generation, organism, target_phrase):
        if self.logging:
            final_organism_log = {
                "type": "final_organism",
                "generation": generation,
                "organism_encoding": organism,
                "decoded_organism": target_phrase,
            }
            self.logs.append(final_organism_log)

    def individual_logging_fitness(self, generation, population, fitness_scores):
        if self.logging and self.individual_logging:
            current_generation_log = self.logs[-1]  # Get the latest generation log
            for index, fitness_score in enumerate(fitness_scores):
                individual_log = {
                    "individual_index": index,
                    "organism": population[index],
                    "fitness_score": fitness_score
                }
                current_generation_log["individuals"].append(individual_log)

    def start_new_generation_logging(self, generation_number):
        generation_log = {
            "generation": generation_number,
            "summary": {},  # Placeholder for summary statistics
            "individuals": [],  # Placeholder for individual fitness logs
            "mutations": [],  # Placeholder for mutation logs
            "crossovers": [],  # Placeholder for logging crossover events
            "organisms": []  # Detailed logging for organisms, including pre- and post-mutation states
        }
        self.logs.append(generation_log)

    def log_new_organism(self, organism_encoding):
        organism_log = {
            "encoding": organism_encoding,
            # Other organism details can go here
        }
        if self.logs:  # Ensure there's at least one generation log
            self.logs[-1]["organisms"].append(organism_log)

    def log_organism_state(self, stage, organism, generation):
        organism_log = {
            "stage": stage,
            "generation": generation,
            "encoded_organism": organism.copy(),
            # Include other relevant details here
        }
        # Append this log to the current generation's log or a dedicated section for organisms
        self.logs[-1]["organisms"].append(organism_log)

    def save_logs(self, logs, file_name=None):
        if file_name is None:
            file_name = f"{self.experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

        # Define the directory path for logs within the current working directory
        logs_dir = os.path.join(os.getcwd(), "logs_and_log_tools")

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # Define the full path for the log file
        full_path = os.path.join(logs_dir, file_name)

        # Save the logs to the file
        with open(full_path, 'w') as f:
            json.dump(logs, f, indent=4)

        print(f"Logs saved to {full_path}")

    # Utility functions

    def initialize_population(self):
        population = []
        for _ in range(int(self.population_size)):
            individual_length = random.randint(2, self.max_individual_length)
            organism = self.encoding_manager.generate_random_organism(functional_length=individual_length,
                                                                      include_specials=self.delimiters,
                                                                      probability=0.10, verbose=False)
            population.append(organism)
        return population

    def decode_organism(self, encoded_organism, format=False):
        encoded_organism = tuple(encoded_organism)
        decoded_genes = self.encoding_manager.decode(encoded_organism, verbose=False)
        if format:
            decoded_genes = [gene for gene in decoded_genes if gene not in ['Start', 'End']]
            return decoded_genes
        return decoded_genes

    def encode_string(self, genetic_string):
        encoded_sequence = []
        for gene in genetic_string:
            if gene in self.encoding_manager.reverse_encodings:
                encoded_gene = self.encoding_manager.reverse_encodings[gene]
                encoded_sequence.append(encoded_gene)
            else:
                print(f"Gene '{gene}' not found in EncodingManager. Adding it now.")
                self.encoding_manager.add_gene(gene)
                encoded_gene = self.encoding_manager.reverse_encodings[gene]
                encoded_sequence.append(encoded_gene)
        return encoded_sequence

    def find_delimited_segments_in_decoded(self, decoded_organism):
        segments = []
        segment_start = None
        for i, gene in enumerate(decoded_organism):
            if gene == 'Start':
                segment_start = i + 1
            elif gene == 'End' and segment_start is not None:
                segments.append((segment_start, i))
                segment_start = None
        return segments

    def validate_delimiters(self, organism, context=""):
        decoded_organism = self.encoding_manager.decode(organism)
        delimiter_stack = []
        for i, gene in enumerate(decoded_organism):
            if gene == 'Start':
                delimiter_stack.append((gene, i))
            elif gene == 'End':
                if not delimiter_stack or delimiter_stack[-1][0] != 'Start':
                    raise ValueError(
                        f"Unmatched 'End' found at index {i} in context '{context}'. Decoded organism: {decoded_organism}")
                delimiter_stack.pop()
        if delimiter_stack:
            unmatched_start = delimiter_stack[-1][1]
            raise ValueError(
                f"Unmatched 'Start' found at index {unmatched_start} in context '{context}'. Decoded organism: {decoded_organism}")
        return organism

    import random

    def select_gene(self, verbose=False):
        # Decide whether to select a base gene or a captured codon
        if random.random() < self.base_gene_prob or not self.encoding_manager.captured_segments:
            # Select a base gene if probability condition is met or if captured_segments is empty
            base_gene = random.choice(self.genes)  # base_gene is now a string representing the gene ID
            if base_gene not in ['Start', 'End']:  # Make sure not to select 'Start' or 'End' as base genes
                gene_key = self.encoding_manager.reverse_encodings[base_gene]
                gene_type = "Base Gene"
            else:
                # If 'Start' or 'End' is randomly selected, choose another gene
                return self.select_gene(verbose)
        else:
            # Select a captured codon with a weighted probability that favors newer genes
            captured_codon_keys = list(self.encoding_manager.captured_segments.keys())
            total_captured = len(captured_codon_keys)
            # Generate weights that decrease exponentially from newer to older genes
            weights = [self.capture_gene_prob ** (total_captured - i - 1) for i in range(total_captured)]
            normalized_weights = [weight / sum(weights) for weight in weights]  # Normalize the weights
            captured_codon_key = random.choices(captured_codon_keys, weights=normalized_weights, k=1)[0]
            gene_key = self.encoding_manager.captured_segments[captured_codon_key]
            gene_type = "Captured Segment"

        if verbose:
            print(f"Selected {gene_type}: {gene_key}")

        return gene_key

    def evaluate_population_fitness(self):
        """Evaluate the fitness of the entire population."""
        if self.before_fitness_evaluation:
            self.before_fitness_evaluation(self)

        self.fitness_scores = self.fitness_evaluator.evaluate(self.population, self)

        if self.after_population_selection:
            self.after_population_selection(self)

        return self.fitness_scores

    # Crossover functions

    def is_fully_delimited(self, organism):
        if not organism:  # Check if organism is empty
            return False  # An empty organism cannot be fully delimited

        start_codon = self.encoding_manager.reverse_encodings['Start']
        end_codon = self.encoding_manager.reverse_encodings['End']
        return organism[0] == start_codon and organism[-1] == end_codon

    def select_and_generate_new_population(self, generation):
        sorted_population = sorted(zip(self.population, self.fitness_scores), key=lambda x: x[1], reverse=True)
        num_elites = int(self.elitism_ratio * self.population_size)
        elites = [individual for individual, _ in sorted_population[:num_elites]]
        new_population = elites[:]

        selected_parents = [individual for individual, _ in sorted_population[:self.num_parents]]
        shift = 0

        while len(new_population) < self.population_size:
            for i in range(0, len(selected_parents) - 1, 2):
                parent1_index = (i + shift) % len(selected_parents)
                parent2_index = (i + 1 + shift) % len(selected_parents)
                parent1, parent2 = selected_parents[parent1_index], selected_parents[parent2_index]

                if self.is_fully_delimited(parent1) or self.is_fully_delimited(parent2):
                    new_population.extend([parent1, parent2][:self.population_size - len(new_population)])
                    continue

                if random.random() < self.crossover_prob:
                    non_delimited_indices = self.get_non_delimiter_indices(parent1, parent2)
                    offspring1, offspring2 = self.crossover(parent1, parent2, non_delimited_indices)
                else:
                    offspring1, offspring2 = parent1[:], parent2[:]

                # Log new offspring before mutation
                self.log_new_organism(offspring1)
                self.log_new_organism(offspring2)

                offspring1 = self.mutate_organism(offspring1, generation)
                offspring2 = self.mutate_organism(offspring2, generation)
                new_population.extend([offspring1, offspring2][:self.population_size - len(new_population)])

            shift += 1



        return new_population

    def process_or_crossover_parents(self, new_population, parent1, parent2, generation):
        if self.is_fully_delimited(parent1) or self.is_fully_delimited(parent2):
            if self.is_fully_delimited(parent1):
                new_population.append(parent1)
            if self.is_fully_delimited(parent2) and len(new_population) < self.population_size:
                new_population.append(parent2)
        else:
            if random.random() < self.crossover_prob:
                non_delimited_indices = self.get_non_delimiter_indices(parent1, parent2)
                offspring1, offspring2 = self.crossover(parent1, parent2, non_delimited_indices)
            else:
                offspring1, offspring2 = parent1[:], parent2[:]

            new_population.extend([self.mutate_organism(offspring1, generation), self.mutate_organism(offspring2,
                                                                                                      generation)][
                                  :self.population_size - len(new_population)])
        return new_population

    def get_non_delimiter_indices(self, parent1, parent2):
        delimiter_indices = self.calculate_delimiter_indices(parent1, parent2)
        non_delimited_indices = set(range(min(len(parent1), len(parent2))))

        for start_idx, end_idx in delimiter_indices:
            non_delimited_indices -= set(range(start_idx, end_idx + 1))

        return list(non_delimited_indices)

    def crossover(self, parent1, parent2, non_delimited_indices):
        crossover_point = self.choose_crossover_point(non_delimited_indices)

        if crossover_point is None:
            offspring1, offspring2 = parent1[:], parent2[:]
        else:
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

        self.log_crossover(self.current_generation, parent1, parent2, crossover_point, offspring1, offspring2)
        return offspring1, offspring2

    def choose_crossover_point(self, non_delimited_indices):
        return random.choice(non_delimited_indices) if non_delimited_indices else None

    def calculate_delimiter_indices(self, parent1, parent2):
        delimiter_indices = []
        for parent in [parent1, parent2]:
            starts = [i for i, codon in enumerate(parent) if codon == self.encoding_manager.reverse_encodings['Start']]
            ends = [i for i, codon in enumerate(parent) if codon == self.encoding_manager.reverse_encodings['End']]
            delimiter_indices.extend(zip(starts, ends))
        return delimiter_indices

    def is_entirely_delimited(self, organism, delimiter_indices):
        return delimiter_indices and delimiter_indices[0][0] == 0 and delimiter_indices[-1][1] == len(organism) - 1

    # Mutation functions

    def mutate_organism(self, organism, generation, mutation=None, log_enhanced=False):
        if self.logging and not log_enhanced:
            self.log_organism_state("before_mutation", organism, generation)
        i = 0
        detailed_logs = []  # Initialize detailed logs collection if enhanced logging is used

        while i < len(organism):
            original = organism[:]  # Capture the state before mutation
            if mutation is None:
                mutation_type = self.select_mutation_type(i, organism)
            else:
                mutation_type = mutation

            # Handle the various types of mutations
            if mutation_type != 'None':
                if mutation_type == 'insertion':
                    organism, i = self.perform_insertion(organism, i)
                elif mutation_type == 'point':
                    organism, i = self.perform_point_mutation(organism, i)
                elif mutation_type == 'swap':
                    organism, i = self.perform_swap(organism, i)
                elif mutation_type == 'delimit_delete':
                    organism, i = self.perform_delimit_delete(organism, i)
                elif mutation_type == 'deletion':
                    organism, i = self.perform_deletion(organism, i)
                elif mutation_type == 'capture':
                    organism, i = self.perform_capture(organism, i)
                elif mutation_type == 'open':
                    organism, i = self.perform_open(organism, i, no_delimit=False)
                elif mutation_type == 'open_no_delimit':
                    organism, i = self.perform_open(organism, i, no_delimit=True)
                elif mutation_type == 'insert_delimiter_pair':
                    organism, i = self.insert_delimiter_pair(organism, i)

                # Collect mutation details if enhanced logging is enabled
                if log_enhanced:
                    detailed_logs.append({
                        "generation": generation,
                        "type": mutation_type,
                        "before": original,
                        "after": organism[:],
                        "index": i
                    })

            i = max(0, min(i + 1, len(organism)))

        # Optionally return detailed logs along with the organism if enhanced logging is enabled
        if log_enhanced:
            return organism, detailed_logs
        else:
            return organism

    def select_mutation_type(self, index, organism):
        gene = organism[index]
        start_codon = self.encoding_manager.reverse_encodings['Start']
        end_codon = self.encoding_manager.reverse_encodings['End']

        # Determine if the gene is within a delimited section by calculating the depth
        depth = 0
        for i, codon in enumerate(organism[:index + 1]):
            if codon == start_codon:
                depth += 1
            elif codon == end_codon:
                depth -= 1

        # Special case mutations with independent probabilities
        if depth > 0 and random.random() <= self.capture_mutation_prob:
            return 'capture'
        if gene not in {start_codon, end_codon} and random.random() <= self.open_mutation_prob:
            return 'open_no_delimit' if depth > 0 else 'open'
        if depth == 0 and gene not in {start_codon, end_codon} and random.random() <= self.delimiter_insert_prob:
            return 'insert_delimiter_pair'

        # Select mutation based on context
        mutation_prob = self.delimited_mutation_prob if depth > 0 else self.mutation_prob
        if random.random() > mutation_prob:
            return 'None'  # No mutation occurs

        # Mutation choices based on gene type and depth
        if gene in {start_codon, end_codon}:
            mutation_choices = ['delimit_delete', 'swap', ]
        elif depth > 0:
            mutation_choices = ['swap', 'point', 'insertion', 'deletion']
        else:
            mutation_choices = ['swap', 'point', 'insertion', 'deletion']

        return random.choice(mutation_choices)

    def insert_delimiter_pair(self, organism, index):
        mutation_log = {
            'type': 'insert_delimiter_pair',
            'generation': self.current_generation,
            'index': index,
            'start_codon_inserted': None,
            'end_codon_inserted': None
        }

        start_codon = self.encoding_manager.reverse_encodings['Start']
        end_codon = self.encoding_manager.reverse_encodings['End']

        organism.insert(index, start_codon)
        mutation_log['start_codon_inserted'] = {'codon': start_codon, 'index': index}

        end_delimiter_index = index + 2

        if end_delimiter_index <= len(organism):
            organism.insert(end_delimiter_index, end_codon)
            mutation_log['end_codon_inserted'] = {'codon': end_codon, 'index': end_delimiter_index}
        else:
            organism.append(end_codon)
            mutation_log['end_codon_inserted'] = {'codon': end_codon, 'index': len(organism) - 1}

        if self.logging and self.mutation_logging:
            self.log_mutation(mutation_log)

        return organism, end_delimiter_index

    def perform_delimit_delete(self, organism, index):
        mutation_log = None
        delimiter_pair = self.find_delimiters(organism, index)

        # Check if a valid pair of delimiters was found
        if delimiter_pair is not None:
            start_location, end_location = delimiter_pair

            # Remove only the delimiters while keeping the segment between them
            if start_location + 1 < end_location:  # Ensure there is content between the delimiters
                # Organism with start delimiter removed
                organism = organism[:start_location] + organism[start_location + 1:end_location] + organism[
                                                                                                   end_location + 1:]
            else:
                # If there is no content between the delimiters, remove both
                organism = organism[:start_location] + organism[end_location + 1:]

            mutation_log = {
                'type': 'delimit_delete',
                'generation': self.current_generation,
                'start_location': start_location,
                'end_location': end_location
            }
            index = start_location  # Adjust index after deletion

        if self.logging and self.mutation_logging and mutation_log is not None:
            self.log_mutation(mutation_log)

        return organism, index

    def perform_insertion(self, organism, index):
        mutation_log = None
        gene_key = self.select_gene()  # This now returns the hash key directly

        # The gene_key is now a hash key, so we can use it directly without conversion
        gene = self.encoding_manager.encodings.get(gene_key, "Unknown")

        organism.insert(index, gene_key)  # Insert the hash key directly into the organism

        mutation_log = {
            'type': 'insertion',
            'generation': self.current_generation,
            'index': index,
            'gene_inserted': gene,
            'codon_inserted': gene_key  # Now using the hash key directly
        }

        if self.logging and self.mutation_logging:
            self.log_mutation(mutation_log)

        return organism, index + 1

    def perform_point_mutation(self, organism, index):
        mutation_log = None
        new_codon = self.select_gene()  # This now returns a hash key directly

        original_codon = organism[index]
        organism[index] = new_codon
        gene = self.encoding_manager.encodings.get(new_codon, "Unknown")

        mutation_log = {
            'type': 'point_mutation',
            'generation': self.current_generation,
            'index': index,
            'original_codon': original_codon,
            'new_codon': new_codon,
            'gene': gene
        }

        if self.logging and self.mutation_logging:
            self.log_mutation(mutation_log)

        return organism, index

    def perform_swap(self, organism, index):
        mutation_log = None
        swap_actions = ['forward', 'backward']
        first_action = random.choice(swap_actions)
        swapped = False

        if first_action == 'forward' and self.can_swap(organism, index, index + 1):
            organism[index], organism[index + 1] = organism[index + 1], organism[index]

            swapped_index = index + 1
            swapped = True
        elif self.can_swap(organism, index, index - 1):
            organism[index], organism[index - 1] = organism[index - 1], organism[index]
            swapped_index = index - 1
            swapped = True

        if swapped:
            mutation_log = {
                'type': 'swap',
                'generation': self.current_generation,
                'index': index,
                'swapped_with_index': swapped_index,
                'original_codon': organism[index],
                'swapped_codon': organism[swapped_index]
            }
        if self.logging and self.mutation_logging:
            self.log_mutation(mutation_log)

        return organism, index

    '''def can_swap(self, organism, index_a, index_b):
        if 0 <= index_a < len(organism) and 0 <= index_b < len(organism):
            return True  # Assuming swapping between valid indices is always allowed
        return False'''

    def can_swap(self, organism, index_a, index_b):
        # Check if indices are within the bounds of the organism
        if 0 <= index_a < len(organism) and 0 <= index_b < len(organism):
            # Access the reverse encoding for 'Start' and 'End'
            start_encoding = self.encoding_manager.reverse_encodings['Start']
            end_encoding = self.encoding_manager.reverse_encodings['End']

            # Check if the genes at index_a or index_b are encoded as 'Start' or 'End'
            if organism[index_a] in [start_encoding, end_encoding] and organism[index_b] in [start_encoding,
                                                                                             end_encoding]:
                return False  # Do not allow swap both genes are delimiters 'Start' or 'End'

            return True

        return False  # Do not allow swap if indices are out of bounds

    def perform_deletion(self, organism, index):
        mutation_log = None
        if len(organism) > 1:
            deleted_codon = organism[index]
            del organism[index]
            index = max(0, index - 1)

            mutation_log = {
                'type': 'deletion',
                'generation': self.current_generation,
                'index': index,
                'deleted_codon': deleted_codon
            }

        if self.logging and self.mutation_logging:
            self.log_mutation(mutation_log)

        return organism, index

    def find_delimiters(self, organism, index):
        start_codon = self.encoding_manager.reverse_encodings['Start']
        end_codon = self.encoding_manager.reverse_encodings['End']
        start_index, end_index = None, None

        # Search backwards from the current index for the nearest start delimiter
        for i in range(index, -1, -1):
            if organism[i] == start_codon:
                start_index = i
                break

        # If a start delimiter is found, search forward for the nearest end delimiter
        if start_index is not None:
            for i in range(start_index + 1, len(organism)):
                if organism[i] == end_codon:
                    end_index = i
                    break

        # If both start and end delimiters are found, return their indices
        if start_index is not None and end_index is not None:
            return start_index, end_index

        # If no valid pair of delimiters is found, return None explicitly
        return None

    def perform_capture(self, organism, index):
        mutation_log = None
        delimiters = self.find_delimiters(organism, index)

        # Proceed only if a valid delimiter pair is found and the segment size meets the criteria
        if delimiters is not None:
            start_index, end_index = delimiters
            segment_size = end_index - start_index - 1  # -1 to exclude delimiters

            # Add a condition to check the segment size
            if segment_size > 1:  # or any other minimum size you consider appropriate
                segment_to_capture = organism[start_index + 1:end_index]

                # Perform the capture operation
                captured_codon = self.encoding_manager.capture_segment(segment_to_capture)
                if captured_codon is not False:
                    # Replace the delimited segment including the delimiters with the captured codon
                    organism = organism[:start_index] + [captured_codon] + organism[end_index + 1:]
                    mutation_log = {
                        'type': 'capture',
                        'generation': self.current_generation,
                        'index': start_index,
                        'captured_segment': segment_to_capture,
                        'captured_codon': captured_codon,
                    }
        if self.logging and self.mutation_logging:
            self.log_mutation(mutation_log)

        return organism, index

    def perform_open(self, organism, index, no_delimit=False):
        mutation_log = None
        decompressed = self.encoding_manager.open_segment(organism[index], no_delimit=no_delimit)
        if decompressed is not False:
            organism = organism[:index] + decompressed + organism[index + 1:]
            index += len(decompressed) - 1  # Adjust the index for the next operation

            mutation_log = {
                'type': 'open',
                'generation': self.current_generation,
                'index': index,
                'opened_codon': organism[index],
                'decompressed_content': decompressed
            }

        if self.logging and self.mutation_logging:
            self.log_mutation(mutation_log)

        return organism, index

    def repair(self, organism):
        start_codon = self.encoding_manager.reverse_encodings['Start']
        end_codon = self.encoding_manager.reverse_encodings['End']
        depth = 0
        last_start_index = -1

        i = 0
        while i < len(organism):
            if organism[i] == start_codon:
                depth += 1
                last_start_index = i
                i += 1
            elif organism[i] == end_codon:
                if depth > 0:
                    depth -= 1
                    last_start_index = -1
                    i += 1
                else:
                    del organism[i]  # Remove unpaired 'End'
            else:
                i += 1

        if depth > 0 and last_start_index != -1:
            del organism[last_start_index]  # Remove unpaired 'Start'

        return organism

    def run_algorithm(self):
        self.population = self.initialize_population()

        for generation in range(self.max_generations):
            self.current_generation = generation
            self.start_new_generation_logging(self.current_generation)

            # Initialize a new log entry for this generation

            self.current_generation = generation  # Update current generation

            # Initialize a new log entry for this generation
            self.start_new_generation_logging(self.current_generation)

            # Callback before fitness evaluation
            if self.before_fitness_evaluation:
                self.before_fitness_evaluation(self)

            self.fitness_scores = [self.fitness_function(individual, self) for individual in self.population]

            # Log the current generation's details if generation logging is enabled
            if self.logging and self.generation_logging:
                self.log_generation(generation, self.fitness_scores, self.population)

            average_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
            print(f"Generation {generation}: Average Fitness = {average_fitness}")

            # Callback after population selection and before new population generation
            if self.after_population_selection:
                self.after_population_selection(self)

            self.population = self.select_and_generate_new_population(generation)

            # Callback before finalizing the current generation (e.g., for additional logging or analysis)
            if self.before_generation_finalize:
                self.before_generation_finalize(self)

            if self.logging and self.individual_logging:
                self.individual_logging_fitness(generation, self.population, self.fitness_scores)

            # Optionally: check for convergence or stopping criteria

        # Compile final logs including initial configuration and outcomes
        print(self.encoding_manager.encodings)
        if self.logging:
            final_log = {"initial_configuration": {
                "MUTATION_PROB": self.mutation_prob,
                "DELIMITED_MUTATION_PROB": self.delimited_mutation_prob,
                "OPEN_MUTATION_PROB": self.open_mutation_prob,
                "CAPTURE_MUTATION_PROB": self.capture_mutation_prob,
                "DELIMITER_INSERT_PROB": self.delimiter_insert_prob,
                "CROSSOVER_PROB": self.crossover_prob,
                "ELITISM_RATIO": self.elitism_ratio,
                "BASE_GENE_PROB": self.base_gene_prob,
                "MAX_INDIVIDUAL_LENGTH": self.max_individual_length,
                "POPULATION_SIZE": self.population_size,
                "NUM_PARENTS": self.num_parents,
                "MAX_GENERATIONS": self.max_generations,
                "DELIMITERS": self.delimiters,
                "DELIMITER_SPACE": self.delimiter_space,
                "seed": self.seed

            },
                # Include other sections of the final_log as needed

                "final_population": self.population,
                "final_fitness_scores": self.fitness_scores,
                "genes": self.genes,
                "final_encodings": self.encoding_manager.encodings,
                "logs": self.logs
            }

            # Save final logs to a specified directory, ensuring the directory exists
            log_folder = "logs_and_log_tools"  # Directory name for logs
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)

            log_filename = f"{log_folder}/{self.experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            with open(log_filename, 'w') as log_file:
                json.dump(final_log, log_file, indent=4)
