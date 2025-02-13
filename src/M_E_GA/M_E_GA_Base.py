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
from .M_E_Engine import EncodingManager
from .GA_Logger import GA_Logger 


class M_E_GA_Base:
    """
    Base class for the genetic algorithm engine.

    This class implements methods to initialize and evolve a population of organisms
    using genetic algorithm operations such as mutation, crossover, and selection.
    It also provides extensive logging facilities for monitoring evolution events.

    Parameters:
        genes (list): List of available genes for organism encoding.
        fitness_function (callable): Function used to evaluate an organism's fitness.
        mutation_prob (float): Base probability for point mutations.
        delimited_mutation_prob (float): Mutation probability for genes within delimiters.
        delimit_delete_prob (float): Probability of deleting a delimiter.
        open_mutation_prob (float): Probability of performing an open mutation.
        metagene_mutation_prob (float): Probability of a metagene mutation.
        delimiter_insert_prob (float): Probability of inserting a delimiter pair.
        crossover_prob (float): Probability of performing a crossover event.
        elitism_ratio (float): Ratio of the population preserved as elites.
        base_gene_prob (float): Probability of selecting a base gene.
        max_individual_length (int): Maximum allowed length for an individual organism.
        population_size (int): Size of the population.
        num_parents (int): Number of parents selected for reproduction.
        max_generations (int): Maximum number of generations to run the algorithm.
        delimiters (bool): Whether to include delimiters in the organism encoding.
        delimiter_space (int): Spacing parameter for delimiters.
        logging (bool): Flag to enable or disable logging.
        generation_logging (bool): Flag to enable logging for each generation.
        mutation_logging (bool): Flag to enable detailed logging for mutations.
        crossover_logging (bool): Flag to enable detailed logging for crossovers.
        individual_logging (bool): Flag to enable logging of individual organisms.
        experiment_name (str): Name of the experiment.
        encodings (dict): Optional pre-defined encodings.
        seed (int): Seed value for random number generator (for reproducibility).
        before_fitness_evaluation (callable): Function to run before fitness evaluation.
        after_population_selection (callable): Function to run after population selection.
        before_generation_finalize (callable): Function to run before finalizing a generation.
        metagene_prob (float): Additional probability for metagene operations.
        fitness_evaluator (object): Instance responsible for evaluating organism fitness.
        lru_cache_size (int): Explicit parameter for the LRU cache size.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, genes, fitness_function,
                 mutation_prob=0.01, delimited_mutation_prob=0.01,
                 delimit_delete_prob=0.01, open_mutation_prob=0.0001,
                 metagene_mutation_prob=0.00001,
                 delimiter_insert_prob=0.00001, crossover_prob=0.50,
                 elitism_ratio=0.06, base_gene_prob=0.98,
                 max_individual_length=6, population_size=400,
                 num_parents=80, max_generations=1000,
                 delimiters=True, delimiter_space=3, logging=True,
                 generation_logging=True, mutation_logging=False,
                 crossover_logging=False, individual_logging=False,
                 experiment_name=None, encodings=None, seed=None,
                 before_fitness_evaluation=None, after_population_selection=None,
                 before_generation_finalize=None, metagene_prob=0.0,
                 fitness_evaluator=None,
                 lru_cache_size=100,   # <<-- New explicit parameter for LRU cache size
                 **kwargs):
        """
        Initialize the genetic algorithm with configuration parameters.

        This constructor sets up the genetic algorithm configuration, seeds the random
        number generator if a seed is provided, and initializes the encoding manager
        and logging facilities.

        Parameters:
            genes (list): List of genes available for encoding.
            fitness_function (callable): Function to evaluate fitness of an organism.
            mutation_prob (float): Base mutation probability for genes.
            delimited_mutation_prob (float): Mutation probability for genes within delimiters.
            delimit_delete_prob (float): Probability of deleting delimiters.
            open_mutation_prob (float): Probability for open mutations.
            metagene_mutation_prob (float): Probability for metagene mutations.
            delimiter_insert_prob (float): Probability for inserting delimiter pairs.
            crossover_prob (float): Probability for performing a crossover.
            elitism_ratio (float): Ratio of elites preserved during selection.
            base_gene_prob (float): Probability of selecting a base gene.
            max_individual_length (int): Maximum length of an individual.
            population_size (int): Number of individuals in the population.
            num_parents (int): Number of parents chosen for reproduction.
            max_generations (int): Maximum number of generations to run.
            delimiters (bool): Flag to include delimiters in organisms.
            delimiter_space (int): Spacing for delimiters.
            logging (bool): Enable or disable logging.
            generation_logging (bool): Enable logging per generation.
            mutation_logging (bool): Enable detailed mutation logging.
            crossover_logging (bool): Enable detailed crossover logging.
            individual_logging (bool): Enable logging of individual organism details.
            experiment_name (str): Name of the experiment.
            encodings (dict): Pre-defined encodings (if any).
            seed (int): Seed for the random number generator.
            before_fitness_evaluation (callable): Callback before fitness evaluation.
            after_population_selection (callable): Callback after population selection.
            before_generation_finalize (callable): Callback before finalizing a generation.
            metagene_prob (float): Probability for metagene operations.
            fitness_evaluator (object): Instance to evaluate organism fitness.
            lru_cache_size (int): Size for the LRU cache in the encoding manager.
            **kwargs: Additional keyword arguments.
        """
        self.genes = genes
        self.fitness_function = fitness_function
        self.fitness_evaluator = fitness_evaluator  # Store the fitness evaluator instance
        self.logging = logging
        self.logs = []
        self.log_filename = ""
        self.experiment_name = experiment_name
        self.before_fitness_evaluation = before_fitness_evaluation
        self.after_population_selection = after_population_selection
        self.before_generation_finalize = before_generation_finalize

        # Set configuration parameters
        self.mutation_prob = mutation_prob
        self.delimited_mutation_prob = delimited_mutation_prob
        self.delimit_delete_prob = delimit_delete_prob
        self.open_mutation_prob = open_mutation_prob
        self.metagene_mutation_prob = metagene_mutation_prob
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
        self.metagene_prob = metagene_prob
        self.fitness_scores = []  # Added to track fitness scores

        # Set the LRU cache size explicitly.
        self.lru_cache_size = lru_cache_size

        # Seed used for reproducibility.
        if seed is not None:
            random.seed(seed)

        # --- New Logging Setup ---
        if self.logging:
            if self.experiment_name is None:
                self.experiment_name = input("Enter the experiment name: ")
            self.log_filename = f"{self.experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            self.logger = GA_Logger(self.experiment_name)
        else:
            self.logger = None

        # Integrate encodings if provided, and add genes to the encoding manager.
        if encodings:
            self.encoding_manager = EncodingManager(lru_cache_size=self.lru_cache_size, logger=self.logger)
            self.encoding_manager.integrate_uploaded_encodings(encodings, self.genes)
        else:
            self.encoding_manager = EncodingManager(lru_cache_size=self.lru_cache_size, logger=self.logger)
            for gene in self.genes:
                self.encoding_manager.add_gene(gene, verbose=True)

    def log_generation(self, generation, fitness_scores, population=None):
        """
        Log summary statistics for a generation.

        This function calculates and logs the average, median, best, and worst
        fitness values for the current generation.

        Parameters:
            generation (int): The current generation number.
            fitness_scores (list): List of fitness scores for the current population.
            population (list, optional): The current population of organisms.
        """
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
            # --- New real-time event logging ---
            if self.logger:
                self.logger.log_event("generation_summary", summary_log)
            current_generation_log = self.logs[-1]  # Assuming this is called at the end of each generation
            current_generation_log["summary"] = summary_log

    def log_mutation(self, mutation_details):
        """
        Log a mutation event.

        Parameters:
            mutation_details (dict): Dictionary containing mutation type, index,
                                     generation, and other related details.
        """
        if self.logging and self.mutation_logging:
            if self.logger:
                self.logger.log_event("mutation", mutation_details)
            if self.logs:
                current_generation_log = self.logs[-1]
                current_generation_log["mutations"].append(mutation_details)

    def log_crossover(self, generation, parent1, parent2, crossover_point, offspring1, offspring2):
        """
        Log a crossover event.

        Parameters:
            generation (int): The current generation number.
            parent1 (list): The first parent organism.
            parent2 (list): The second parent organism.
            crossover_point (int or None): The index at which crossover occurs.
            offspring1 (list): The first offspring produced.
            offspring2 (list): The second offspring produced.
        """
        if self.logging and self.crossover_logging:
            crossover_log = {
                "crossover_point": crossover_point,
                "parent1_before": parent1[:crossover_point] if crossover_point is not None else parent1,
                "parent2_before": parent2[:crossover_point] if crossover_point is not None else parent2,
                "offspring1": offspring1,
                "offspring2": offspring2,
            }
            if self.logger:
                self.logger.log_event("crossover", crossover_log)
            current_generation_log = self.logs[-1]
            current_generation_log["crossovers"].append(crossover_log)

    def log_fitness_function_settings(self, settings):
        """
        Log the settings used for the fitness function.

        This function logs settings such as volume and size constraints if applicable.

        Parameters:
            settings (dict): A dictionary of settings for the fitness function.
        """
        if self.logging and self.fitness_settings_logging and not self.fitness_settings_logged:
            settings.update({
                "MAX_VOLUME": self.max_volume,
                "VOLUME_PENALTY_FACTOR": self.volume_penalty_factor,
                "MAX_SIZE": self.max_size,
                "SIZE_PENALTY_FACTOR": self.size_penalty_factor
            })
            self.logs.append({"fitness_function_settings": settings})
            self.fitness_settings_logged = True

    def log_final_organism(self, generation, organism, target_phrase):
        """
        Log the final organism after algorithm completion.

        Parameters:
            generation (int): The generation number at which the final organism was produced.
            organism (list): The encoded final organism.
            target_phrase (str): The decoded representation or target phrase.
        """
        if self.logging:
            final_organism_log = {
                "type": "final_organism",
                "generation": generation,
                "organism_encoding": organism,
                "decoded_organism": target_phrase,
            }
            self.logs.append(final_organism_log)

    def individual_logging_fitness(self, generation, population, fitness_scores):
        """
        Log fitness scores for each individual organism.

        Parameters:
            generation (int): The current generation number.
            population (list): List of organism encodings.
            fitness_scores (list): List of fitness scores corresponding to the population.
        """
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
        """
        Initialize logging for a new generation.

        Parameters:
            generation_number (int): The number of the new generation.
        """
        generation_log = {
            "generation": generation_number,
            "summary": {},           # Placeholder for summary statistics
            "individuals": [],       # Placeholder for individual fitness logs
            "mutations": [],         # Placeholder for mutation logs
            "crossovers": [],        # Placeholder for logging crossover events
            "organisms": []          # Detailed logging for organisms, including pre- and post-mutation states
        }
        self.logs.append(generation_log)

    def log_new_organism(self, organism_encoding):
        """
        Log a new organism's encoding.

        Parameters:
            organism_encoding (list): The encoding of the new organism.
        """
        organism_log = {
            "encoding": organism_encoding,
            # Other organism details can go here
        }
        if self.logs:
            self.logs[-1]["organisms"].append(organism_log)

    def log_organism_state(self, stage, organism, generation):
        """
        Log the state of an organism at a given stage.

        Parameters:
            stage (str): The stage of processing (e.g., "before_mutation").
            organism (list): The organism's encoding.
            generation (int): The generation number.
        """
        organism_log = {
            "stage": stage,
            "generation": generation,
            "encoded_organism": organism.copy(),
        }
        self.logs[-1]["organisms"].append(organism_log)

    def save_logs(self, logs, file_name=None):
        """
        Save the logs to a JSON file.

        Parameters:
            logs (list): The logs data to be saved.
            file_name (str, optional): The filename to use. If None, a default name is generated.
        """
        if file_name is None:
            file_name = f"{self.experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        logs_dir = os.path.join(os.getcwd(), "logs_and_log_tools")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        full_path = os.path.join(logs_dir, file_name)
        with open(full_path, 'w') as f:
            json.dump(logs, f, indent=4)
        print(f"Logs saved to {full_path}")

    def initialize_population(self):
        """
        Generate an initial population of organisms.

        Returns:
            list: A list of organism encodings representing the initial population.
        """
        population = []
        for _ in range(int(self.population_size)):
            individual_length = random.randint(2, self.max_individual_length)
            organism = self.encoding_manager.generate_random_organism(functional_length=individual_length,
                                                                      include_specials=self.delimiters,
                                                                      probability=0.10, verbose=False)
            population.append(organism)
        return population

    def decode_organism(self, encoded_organism, format=False):
        """
        Decode an encoded organism into its gene representation.

        Parameters:
            encoded_organism (iterable): The encoded organism.
            format (bool): If True, remove special markers like 'Start' and 'End'.

        Returns:
            list: The list of decoded genes.
        """
        encoded_organism = tuple(encoded_organism)
        decoded_genes = self.encoding_manager.decode(encoded_organism, verbose=False)
        if format:
            decoded_genes = [gene for gene in decoded_genes if gene not in ['Start', 'End']]
            return decoded_genes
        return decoded_genes

    def encode_string(self, genetic_string):
        """
        Encode a string of genes into its corresponding genetic code.

        Parameters:
            genetic_string (iterable): Sequence of gene characters.

        Returns:
            list: The encoded gene sequence.
        """
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
        """
        Identify segments in a decoded organism that are delimited by special markers.

        Parameters:
            decoded_organism (list): The decoded organism representation.

        Returns:
            list: A list of tuples, where each tuple contains the start and end indices of a delimited segment.
        """
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
        """
        Validate that the delimiters in an organism are correctly matched.

        Parameters:
            organism (list): The encoded organism.
            context (str): Optional context for error messages.

        Returns:
            list: The validated organism.

        Raises:
            ValueError: If unmatched delimiters are found.
        """
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

    def select_gene(self, verbose=False):
        """
        Select a gene for mutation or insertion.

        Depending on a random choice, a base gene or a meta gene is selected.

        Parameters:
            verbose (bool): If True, print details of the selection.

        Returns:
            The selected gene key.
        """
        if random.random() < self.base_gene_prob or not self.encoding_manager.meta_genes:
            base_gene = random.choice(self.genes)
            if base_gene not in ['Start', 'End']:
                gene_key = self.encoding_manager.reverse_encodings[base_gene]
                gene_type = "Base Gene"
            else:
                return self.select_gene(verbose=verbose)
        else:
            meta_gene_keys = self.encoding_manager.meta_gene_stack
            total_meta = len(meta_gene_keys)
            weights = [self.metagene_prob ** (total_meta - i - 1) for i in range(total_meta)]
            weight_sum = sum(weights)
            if weight_sum == 0:
                normalized_weights = [1.0 / total_meta] * total_meta
            else:
                normalized_weights = [w / weight_sum for w in weights]
            meta_gene_key = random.choices(meta_gene_keys, weights=normalized_weights, k=1)[0]
            gene_key = meta_gene_key
            gene_type = "Meta Gene"
        if verbose:
            print(f"Selected {gene_type}: {gene_key}")
        return gene_key

    def evaluate_population_fitness(self):
        """
        Evaluate the fitness of the entire population.

        This method optionally calls a pre-evaluation callback, evaluates the population
        using the fitness evaluator, and then calls a post-evaluation callback.

        Returns:
            list: Fitness scores for the current population.
        """
        if self.before_fitness_evaluation:
            self.before_fitness_evaluation(self)
        self.fitness_scores = self.fitness_evaluator.evaluate(self.population, self)
        if self.after_population_selection:
            self.after_population_selection(self)
        return self.fitness_scores

    def is_fully_delimited(self, organism):
        """
        Check if an organism is fully delimited.

        An organism is fully delimited if it starts with a 'Start' codon and ends with an 'End' codon.

        Parameters:
            organism (list): The encoded organism.

        Returns:
            bool: True if fully delimited, False otherwise.
        """
        if not organism:
            return False
        start_codon = self.encoding_manager.reverse_encodings['Start']
        end_codon = self.encoding_manager.reverse_encodings['End']
        return organism[0] == start_codon and organism[-1] == end_codon

    def select_and_generate_new_population(self, generation):
        """
        Select individuals and generate a new population through reproduction.

        The method implements elitism, selection of parents, crossover, and mutation
        to generate the next generation.

        Parameters:
            generation (int): The current generation number.

        Returns:
            list: The new population of organisms.
        """
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
                self.log_new_organism(offspring1)
                self.log_new_organism(offspring2)
                offspring1 = self.mutate_organism(offspring1, generation)
                offspring2 = self.mutate_organism(offspring2, generation)
                new_population.extend([offspring1, offspring2][:self.population_size - len(new_population)])
            shift += 1
        return new_population

    def process_or_crossover_parents(self, new_population, parent1, parent2, generation):
        """
        Process two parent organisms by either copying them directly if delimited or
        performing crossover and mutation.

        Parameters:
            new_population (list): The list representing the new population.
            parent1 (list): The first parent organism.
            parent2 (list): The second parent organism.
            generation (int): The current generation number.

        Returns:
            list: Updated population including offspring from the parents.
        """
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
            new_population.extend([self.mutate_organism(offspring1, generation), self.mutate_organism(offspring2, generation)][
                                  :self.population_size - len(new_population)])
        return new_population

    def get_non_delimiter_indices(self, parent1, parent2):
        """
        Determine the indices that are not part of delimited segments for crossover.

        Parameters:
            parent1 (list): The first parent organism.
            parent2 (list): The second parent organism.

        Returns:
            list: Indices that are safe for crossover operations.
        """
        delimiter_indices = self.calculate_delimiter_indices(parent1, parent2)
        non_delimited_indices = set(range(min(len(parent1), len(parent2))))
        for start_idx, end_idx in delimiter_indices:
            non_delimited_indices -= set(range(start_idx, end_idx + 1))
        return list(non_delimited_indices)

    def crossover(self, parent1, parent2, non_delimited_indices):
        """
        Perform a crossover operation between two parents.

        Parameters:
            parent1 (list): The first parent organism.
            parent2 (list): The second parent organism.
            non_delimited_indices (list): Indices allowed for crossover.

        Returns:
            tuple: Two offspring resulting from the crossover.
        """
        crossover_point = self.choose_crossover_point(non_delimited_indices)
        if crossover_point is None:
            offspring1, offspring2 = parent1[:], parent2[:]
        else:
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        self.log_crossover(self.current_generation, parent1, parent2, crossover_point, offspring1, offspring2)
        return offspring1, offspring2

    def choose_crossover_point(self, non_delimited_indices):
        """
        Randomly select a crossover point from non-delimited indices.

        Parameters:
            non_delimited_indices (list): List of indices not in delimited segments.

        Returns:
            int or None: The selected crossover index, or None if no valid index exists.
        """
        return random.choice(non_delimited_indices) if non_delimited_indices else None

    def calculate_delimiter_indices(self, parent1, parent2):
        """
        Calculate the indices of delimiter segments for both parents.

        Parameters:
            parent1 (list): The first parent organism.
            parent2 (list): The second parent organism.

        Returns:
            list: A list of tuples indicating the start and end indices of delimiter segments.
        """
        delimiter_indices = []
        for parent in [parent1, parent2]:
            starts = [i for i, codon in enumerate(parent) if codon == self.encoding_manager.reverse_encodings['Start']]
            ends = [i for i, codon in enumerate(parent) if codon == self.encoding_manager.reverse_encodings['End']]
            delimiter_indices.extend(zip(starts, ends))
        return delimiter_indices

    def is_entirely_delimited(self, organism, delimiter_indices):
        """
        Determine if the organism is completely enclosed by delimiters.

        Parameters:
            organism (list): The encoded organism.
            delimiter_indices (list): A list of delimiter index tuples.

        Returns:
            bool: True if the entire organism is delimited, False otherwise.
        """
        return delimiter_indices and delimiter_indices[0][0] == 0 and delimiter_indices[-1][1] == len(organism) - 1

    def mutate_organism(self, organism, generation, mutation=None, log_enhanced=False):
        """
        Mutate an organism by applying various mutation operations.

        Iterates through the organism and applies a mutation based on the mutation probability,
        type, and depth (inside or outside delimiters).

        Parameters:
            organism (list): The encoded organism to mutate.
            generation (int): The current generation number.
            mutation (optional): Specific mutation parameter (not used in current implementation).
            log_enhanced (bool): If True, return detailed mutation logs.

        Returns:
            list: The mutated organism.
            OR
            tuple: (mutated organism, detailed_logs) if log_enhanced is True.
        """
        if self.logging and not log_enhanced:
            self.log_organism_state("before_mutation", organism, generation)
        i = 0
        detailed_logs = []
        while i < len(organism):
            original = organism[:]
            depth = self.calculate_depth(organism, i)
            gene = organism[i]
            start_codon = self.encoding_manager.reverse_encodings['Start']
            end_codon = self.encoding_manager.reverse_encodings['End']
            if depth > 0:
                mutation_prob = self.delimited_mutation_prob
            else:
                mutation_prob = self.mutation_prob
            if random.random() <= mutation_prob:
                mutation_type = self.select_mutation_type(i, organism, depth)
                organism, i = self.apply_mutation(organism, i, mutation_type)
                if log_enhanced:
                    detailed_logs.append({
                        "generation": generation,
                        "type": mutation_type,
                        "before": original,
                        "after": organism[:],
                        "index": i
                    })
            else:
                i += 1
        if log_enhanced:
            return organism, detailed_logs
        else:
            return organism

    def select_mutation_type(self, index, organism, depth):
        """
        Select the type of mutation to perform based on the gene and its context.

        Parameters:
            index (int): The current index in the organism.
            organism (list): The encoded organism.
            depth (int): The nesting depth (e.g., within delimiters).

        Returns:
            str: The selected mutation type.
        """
        gene = organism[index]
        start_codon = self.encoding_manager.reverse_encodings['Start']
        end_codon = self.encoding_manager.reverse_encodings['End']
        mutation_choices = []
        mutation_weights = []
        if gene in {start_codon, end_codon}:
            if random.random() < self.delimit_delete_prob:
                mutation_choices = ['delimit_delete']
                mutation_weights = [1.0]
            else:
                mutation_choices = ['swap']
                mutation_weights = [1.0]
        else:
            if depth > 0:
                mutation_choices = ['point', 'swap', 'insertion', 'deletion', 'capture', 'open_no_delimit']
                mutation_weights = [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    self.metagene_mutation_prob,
                    self.open_mutation_prob
                ]
            else:
                mutation_choices = ['point', 'swap', 'insertion', 'deletion', 'insert_delimiter_pair', 'open']
                mutation_weights = [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    self.delimiter_insert_prob,
                    self.open_mutation_prob
                ]
        if not gene in {start_codon, end_codon}:
            total_weight = sum(mutation_weights)
            normalized_probs = [w / total_weight for w in mutation_weights]
            mutation_type = random.choices(mutation_choices, weights=normalized_probs, k=1)[0]
        else:
            mutation_type = random.choices(mutation_choices, weights=mutation_weights, k=1)[0]
        return mutation_type

    def apply_mutation(self, organism, index, mutation_type):
        """
        Apply the selected mutation operation on the organism.

        Parameters:
            organism (list): The encoded organism.
            index (int): The index at which the mutation is applied.
            mutation_type (str): The type of mutation to apply.

        Returns:
            tuple: The mutated organism and the updated index.
        """
        if mutation_type == 'insertion':
            organism, index = self.perform_insertion(organism, index)
        elif mutation_type == 'point':
            organism, index = self.perform_point_mutation(organism, index)
        elif mutation_type == 'swap':
            organism, index = self.perform_swap(organism, index)
        elif mutation_type == 'delimit_delete':
            organism, index = self.perform_delimit_delete(organism, index)
        elif mutation_type == 'deletion':
            organism, index = self.perform_deletion(organism, index)
        elif mutation_type == 'capture':
            organism, index = self.perform_capture(organism, index)
        elif mutation_type == 'open':
            organism, index = self.perform_open(organism, index, no_delimit=False)
        elif mutation_type == 'open_no_delimit':
            organism, index = self.perform_open(organism, index, no_delimit=True)
        elif mutation_type == 'insert_delimiter_pair':
            organism, index = self.insert_delimiter_pair(organism, index)
        else:
            index += 1
        return organism, index

    def calculate_depth(self, organism, index):
        """
        Calculate the nesting depth (i.e., how many delimiters are open) at a given index.

        Parameters:
            organism (list): The encoded organism.
            index (int): The index at which to calculate depth.

        Returns:
            int: The current depth (number of open delimiters).
        """
        start_codon = self.encoding_manager.reverse_encodings['Start']
        end_codon = self.encoding_manager.reverse_encodings['End']
        depth = 0
        for codon in organism[:index + 1]:
            if codon == start_codon:
                depth += 1
            elif codon == end_codon:
                depth -= 1
        return depth

    def insert_delimiter_pair(self, organism, index):
        """
        Insert a pair of delimiter codons into the organism at the specified index.

        Parameters:
            organism (list): The encoded organism.
            index (int): The index at which to insert the delimiter pair.

        Returns:
            tuple: The modified organism and the index position after insertion.
        """
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
        """
        Delete a segment enclosed by delimiters if possible.

        Parameters:
            organism (list): The encoded organism.
            index (int): The current index at which deletion is considered.

        Returns:
            tuple: The organism after deletion and the updated index.
        """
        mutation_log = None
        delimiter_pair = self.find_delimiters(organism, index)
        if delimiter_pair is not None:
            start_location, end_location = delimiter_pair
            if start_location + 1 < end_location:
                organism = organism[:start_location] + organism[start_location + 1:end_location] + organism[end_location + 1:]
            else:
                organism = organism[:start_location] + organism[end_location + 1:]
            mutation_log = {
                'type': 'delimit_delete',
                'generation': self.current_generation,
                'start_location': start_location,
                'end_location': end_location
            }
            index = start_location
        if self.logging and self.mutation_logging and mutation_log is not None:
            self.log_mutation(mutation_log)
        return organism, index

    def perform_insertion(self, organism, index):
        """
        Insert a new gene into the organism at the specified index.

        Parameters:
            organism (list): The encoded organism.
            index (int): The index at which to insert the new gene.

        Returns:
            tuple: The organism after insertion and the updated index.
        """
        mutation_log = None
        gene_key = self.select_gene()
        gene = self.encoding_manager.encodings.get(gene_key, "Unknown")
        organism.insert(index, gene_key)
        mutation_log = {
            'type': 'insertion',
            'generation': self.current_generation,
            'index': index,
            'gene_inserted': gene,
            'codon_inserted': gene_key
        }
        if self.logging and self.mutation_logging:
            self.log_mutation(mutation_log)
        return organism, index + 1

    def perform_point_mutation(self, organism, index):
        """
        Perform a point mutation on the organism at the specified index.

        Parameters:
            organism (list): The encoded organism.
            index (int): The index at which the point mutation is applied.

        Returns:
            tuple: The organism after mutation and the unchanged index.
        """
        mutation_log = None
        new_codon = self.select_gene()
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
        """
        Swap the gene at the specified index with one of its neighbors.

        Parameters:
            organism (list): The encoded organism.
            index (int): The index of the gene to swap.

        Returns:
            tuple: The organism after swapping and the index.
        """
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

    def can_swap(self, organism, index_a, index_b):
        """
        Check whether two positions in the organism can be swapped.

        Parameters:
            organism (list): The encoded organism.
            index_a (int): The first index.
            index_b (int): The second index.

        Returns:
            bool: True if swapping is allowed, False otherwise.
        """
        if 0 <= index_a < len(organism) and 0 <= index_b < len(organism):
            start_encoding = self.encoding_manager.reverse_encodings['Start']
            end_encoding = self.encoding_manager.reverse_encodings['End']
            if organism[index_a] in [start_encoding, end_encoding] and organism[index_b] in [start_encoding, end_encoding]:
                return False
            return True
        return False

    def perform_deletion(self, organism, index):
        """
        Delete the gene at the specified index from the organism.

        Parameters:
            organism (list): The encoded organism.
            index (int): The index of the gene to delete.

        Returns:
            tuple: The organism after deletion and the updated index.
        """
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
        """
        Find the closest pair of delimiters surrounding the given index.

        Parameters:
            organism (list): The encoded organism.
            index (int): The index from which to search for delimiters.

        Returns:
            tuple or None: (start_index, end_index) if a delimiter pair is found; otherwise, None.
        """
        start_codon = self.encoding_manager.reverse_encodings['Start']
        end_codon = self.encoding_manager.reverse_encodings['End']
        start_index, end_index = None, None
        for i in range(index, -1, -1):
            if organism[i] == start_codon:
                start_index = i
                break
        if start_index is not None:
            for i in range(start_index + 1, len(organism)):
                if organism[i] == end_codon:
                    end_index = i
                    break
        if start_index is not None and end_index is not None:
            return start_index, end_index
        return None

    def perform_capture(self, organism, index):
        """
        Perform a capture mutation by compressing a delimited segment into a meta gene.

        Parameters:
            organism (list): The encoded organism.
            index (int): The index at which to perform the capture.

        Returns:
            tuple: The organism after performing the capture and the updated index.
        """
        mutation_log = None
        delimiters = self.find_delimiters(organism, index)
        if delimiters is not None:
            start_index, end_index = delimiters
            segment_size = end_index - start_index - 1
            if segment_size > 1:
                segment_to_capture = organism[start_index + 1:end_index]
                captured_codon = self.encoding_manager.capture_metagene(segment_to_capture)
                if captured_codon is not False:
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
        """
        Perform an open mutation that expands a meta gene back into its full sequence.

        Parameters:
            organism (list): The encoded organism.
            index (int): The index of the meta gene to open.
            no_delimit (bool): If True, open without adding delimiters.

        Returns:
            tuple: The organism after opening the meta gene and the updated index.
        """
        mutation_log = None
        decompressed = self.encoding_manager.open_metagene(organism[index], no_delimit=no_delimit)
        if decompressed is not False:
            organism = organism[:index] + decompressed + organism[index + 1:]
            index += len(decompressed) - 1
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
        """
        Repair an organism by ensuring all delimiters are correctly matched.

        This method iterates through the organism to remove any unmatched
        'Start' or 'End' codons.

        Parameters:
            organism (list): The encoded organism to repair.

        Returns:
            list: The repaired organism.
        """
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
                    del organism[i]
            else:
                i += 1
        if depth > 0 and last_start_index != -1:
            del organism[last_start_index]
        return organism

    def run_algorithm(self):
        """
        Execute the genetic algorithm over a specified number of generations.

        This method orchestrates the entire evolution process, including
        population initialization, fitness evaluation, selection, crossover,
        mutation, and logging at each generation.

        Upon completion, it outputs the final encodings and saves all logs.

        Returns:
            None
        """
        self.population = self.initialize_population()
        for generation in range(self.max_generations):
            self.current_generation = generation
            self.start_new_generation_logging(self.current_generation)
            self.encoding_manager.start_new_generation()
            if self.before_fitness_evaluation:
                self.before_fitness_evaluation(self)
            self.fitness_scores = [self.fitness_function(individual, self) for individual in self.population]
            if self.logging and self.generation_logging:
                self.log_generation(generation, self.fitness_scores, self.population)
            average_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
            print(f"Generation {generation}: Average Fitness = {average_fitness}")
            if self.after_population_selection:
                self.after_population_selection(self)
            self.population = self.select_and_generate_new_population(generation)
            if self.before_generation_finalize:
                self.before_generation_finalize(self)
            if self.logging and self.individual_logging:
                self.individual_logging_fitness(generation, self.population, self.fitness_scores)
        print(self.encoding_manager.encodings)
        if self.logging:
            final_log = {
                "initial_configuration": {
                    "MUTATION_PROB": self.mutation_prob,
                    "DELIMITED_MUTATION_PROB": self.delimited_mutation_prob,
                    "DELIMIT_DELETE_PROB": self.delimit_delete_prob,
                    "OPEN_MUTATION_PROB": self.open_mutation_prob,
                    "CAPTURE_MUTATION_PROB": self.metagene_mutation_prob,
                    "DELIMITER_INSERT_PROB": self.delimiter_insert_prob,
                    "CROSSOVER_PROB": self.crossover_prob,
                    "ELITISM_RATIO": self.elitism_ratio,
                    "BASE_GENE_PROB": self.base_gene_prob,
                    "CAPTURED_GENE_PROB": self.metagene_prob,
                    "MAX_INDIVIDUAL_LENGTH": self.max_individual_length,
                    "POPULATION_SIZE": self.population_size,
                    "NUM_PARENTS": self.num_parents,
                    "MAX_GENERATIONS": self.max_generations,
                    "DELIMITERS": self.delimiters,
                    "DELIMITER_SPACE": self.delimiter_space,
                    "seed": self.seed
                },
                "final_population": self.population,
                "final_fitness_scores": self.fitness_scores,
                "genes": self.genes,
                "final_encodings": self.encoding_manager.encodings,
                "logs": self.logs
            }
            log_folder = "logs_and_log_tools"
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            log_filename = f"{log_folder}/{self.experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            with open(log_filename, 'w') as log_file:
                json.dump(final_log, log_file, indent=4)
            # --- New: Save the real-time logger events as well ---
            if self.logger:
                self.logger.save()
