# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:09:43 2024

@author: Matt Andrews

@file: M_E_GA_Base.py

The central coordinating class for the Genetic Algorithm engine.

"""
import datetime
import json
import os
import random

from .GA_Logger import GA_Logger
from .M_E_Engine import EncodingManager
from .engine.crossover_manager import CrossoverManager
from .engine.logging_manager import LoggingManager
from .engine.mutation_manager import MutationManager
from .engine.population_manager import PopulationManager


class M_E_GA_Base:
    """
    Base class for the genetic algorithm engine. It orchestrates:
     - GA configuration & initialization
     - Population generation and evolutionary loop
     - Delegation to manager classes for specific responsibilities
     - Summarizing final logs

    This class now defers most logging details to the LoggingManager.
    """

    def __init__(
            self,
            genes,
            fitness_function,
            mutation_prob=0.01,
            delimited_mutation_prob=0.01,
            delimit_delete_prob=0.01,
            open_mutation_prob=0.0001,
            metagene_mutation_prob=0.00001,
            delimiter_insert_prob=0.00001,
            crossover_prob=0.50,
            elitism_ratio=0.06,
            base_gene_prob=0.98,
            max_individual_length=6,
            population_size=400,
            num_parents=80,
            max_generations=1000,
            delimiters=True,
            delimiter_space=3,
            logging=True,
            generation_logging=True,
            mutation_logging=False,
            crossover_logging=False,
            individual_logging=False,
            experiment_name=None,
            encodings=None,
            seed=None,
            before_fitness_evaluation=None,
            after_population_selection=None,
            before_generation_finalize=None,
            metagene_prob=0.0,
            fitness_evaluator=None,
            lru_cache_size=100,
            **kwargs
    ):
        """
        Initialize the genetic algorithm with configuration parameters.

        :param genes: A list of available base gene strings.
        :param fitness_function: A callable for evaluating individual fitness
                                 (used if fitness_evaluator is None).
        :param mutation_prob: Base probability for point mutations.
        :param delimited_mutation_prob: Mutation probability for genes inside delimiters.
        :param delimit_delete_prob: Probability of deleting delimiter pairs.
        :param open_mutation_prob: Probability of opening meta-genes.
        :param metagene_mutation_prob: Probability for capturing a segment as a metagene.
        :param delimiter_insert_prob: Probability of inserting a delimiter pair.
        :param crossover_prob: Probability of performing a crossover event.
        :param elitism_ratio: Fraction of population that is carried over as elites.
        :param base_gene_prob: Probability of choosing a base gene vs. a meta gene.
        :param max_individual_length: Maximum length for an individual's encoding.
        :param population_size: The number of individuals in the population.
        :param num_parents: The number of parents used in reproduction.
        :param max_generations: How many generations to run the GA.
        :param delimiters: Whether to include Start/End delimiters in random organisms.
        :param delimiter_space: The spacing for random insertion of delimiters in new organisms.
        :param logging: Enable or disable all logging.
        :param generation_logging: If True, logs generation summaries.
        :param mutation_logging: If True, logs each mutation event in detail.
        :param crossover_logging: If True, logs each crossover event in detail.
        :param individual_logging: If True, logs each individual's fitness each generation.
        :param experiment_name: Name of the experiment (used for logging filenames).
        :param encodings: Optionally supply a pre-built dictionary of encodings.
        :param seed: A random seed for reproducibility.
        :param before_fitness_evaluation: A callable invoked before the population is evaluated.
        :param after_population_selection: A callable invoked after population selection.
        :param before_generation_finalize: A callable invoked before the generation finalizes.
        :param metagene_prob: Additional weighting factor used for meta-gene selection.
        :param fitness_evaluator: An object that handles population-level fitness evaluation
                                  (overrides fitness_function if provided).
        :param lru_cache_size: The size of the LRU cache for metagene usage.
        :param kwargs: Additional arguments that might be used in extended setups.
        """
        self.genes = genes
        self.fitness_function = fitness_function
        self.fitness_evaluator = fitness_evaluator

        # Logging flags
        self.logging = logging
        self.generation_logging = generation_logging
        self.mutation_logging = mutation_logging
        self.crossover_logging = crossover_logging
        self.individual_logging = individual_logging
        self.experiment_name = experiment_name

        self.before_fitness_evaluation = before_fitness_evaluation
        self.after_population_selection = after_population_selection
        self.before_generation_finalize = before_generation_finalize

        # GA config parameters
        self.mutation_prob = mutation_prob
        self.delimited_mutation_prob = delimited_mutation_prob
        self.delimit_delete_prob = delimit_delete_prob
        self.open_mutation_prob = open_mutation_prob
        self.metagene_mutation_prob = metagene_mutation_prob
        self.delimiter_insert_prob = delimiter_insert_prob
        self.crossover_prob = crossover_prob
        self.elitism_ratio = elitism_ratio
        self.base_gene_prob = base_gene_prob
        self.metagene_prob = metagene_prob
        self.max_individual_length = max_individual_length
        self.population_size = population_size
        self.num_parents = num_parents
        self.max_generations = max_generations
        self.delimiters = delimiters
        self.delimiter_space = delimiter_space
        self.seed = seed

        self.population = []
        self.current_generation = 0
        self.fitness_scores = []

        self.lru_cache_size = lru_cache_size

        # Seed the RNG if provided
        if seed is not None:
            random.seed(seed)

        # Setup real-time event logger if logging is on
        if self.logging:
            if self.experiment_name is None:
                # Could prompt or default
                self.experiment_name = "UnnamedExperiment"
            self.logger = GA_Logger(self.experiment_name)
        else:
            self.logger = None

        # Create an EncodingManager, integrate encodings if provided
        self.encoding_manager = EncodingManager(lru_cache_size=self.lru_cache_size, logger=self.logger)
        if encodings:
            self.encoding_manager.integrate_uploaded_encodings(encodings, self.genes)
        else:
            for gene in self.genes:
                self.encoding_manager.add_gene(gene, verbose=True)

        # Instantiate manager classes
        self.population_manager = PopulationManager(self)
        self.mutation_manager = MutationManager(self)
        self.crossover_manager = CrossoverManager(self)

        # Instantiate the LoggingManager, used for generation-level logs
        self.logging_manager = LoggingManager(
            logging_enabled=self.logging,
            generation_logging=self.generation_logging,
            mutation_logging=self.mutation_logging,
            crossover_logging=self.crossover_logging,
            individual_logging=self.individual_logging,
            logger=self.logger
        )

    def decode_organism(self, encoded_organism, format=False):
        """
        Decode an encoded organism into its gene representation.

        :param encoded_organism: The list/tuple of codons (hash keys).
        :param format: If True, remove 'Start'/'End' from the result.
        :return: A list of decoded genes, possibly excluding delimiters if format=True.
        """
        encoded_organism = tuple(encoded_organism)
        decoded_genes = self.encoding_manager.decode(encoded_organism, verbose=False)
        if format:
            decoded_genes = [g for g in decoded_genes if g not in ['Start', 'End']]
        return decoded_genes

    def encode_string(self, genetic_string):
        """
        Encode a sequence of gene strings into their numeric codon representation.

        :param genetic_string: A list of gene symbols to encode.
        :return: A list of integer codons.
        """
        encoded_sequence = []
        for gene in genetic_string:
            if gene in self.encoding_manager.reverse_encodings:
                encoded_sequence.append(self.encoding_manager.reverse_encodings[gene])
            else:
                # Add gene if missing
                self.encoding_manager.add_gene(gene)
                encoded_sequence.append(self.encoding_manager.reverse_encodings[gene])
        return encoded_sequence

    def initialize_population(self):
        """
        Public method to initialize the population using the population manager.
        Useful for advanced usage if you want to manually do an 'init' step.

        :return: A newly generated population (list of organism encodings).
        """
        self.population = self.population_manager.initialize_population()
        return self.population

    def run_algorithm(self):
        """
        Execute the genetic algorithm for max_generations iterations.

        1. Initialize population
        2. For each generation:
            a) Log the generation start
            b) Evaluate fitness
            c) Log generation stats
            d) Generate new population
            e) Possibly log additional individual stats
        3. Dump logs, print final encodings
        """
        # 1. Initialize population if empty
        if not self.population:
            self.population = self.population_manager.initialize_population()

        # 2. Main loop
        for generation in range(self.max_generations):
            self.current_generation = generation
            # Start new generation log
            self.logging_manager.start_new_generation_logging(generation)

            # Start new generation in encoding manager (for LRU usage/deletion)
            self.encoding_manager.start_new_generation()

            # Evaluate fitness
            self.fitness_scores = self.population_manager.evaluate_population_fitness(self.population)

            # Generation summary log
            self.logging_manager.log_generation(generation, self.fitness_scores, self.population)

            # Print short info
            avg_fit = sum(self.fitness_scores) / len(self.fitness_scores)
            print(f"Generation {generation}: Average Fitness = {avg_fit}")

            # Generate next population
            self.population = self.population_manager.select_and_generate_new_population(
                self.population, self.fitness_scores, generation
            )

            # Optional user callback
            if self.before_generation_finalize:
                self.before_generation_finalize(self)

            # If individual logging is on, store each individual's data
            self.logging_manager.individual_logging_fitness(generation, self.population, self.fitness_scores)

        # 3. After finishing all generations
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
                    "seed": self.seed,
                },
                "final_population": self.population,
                "final_fitness_scores": self.fitness_scores,
                "genes": self.genes,
                "final_encodings": self.encoding_manager.encodings,
                "logs": self.logging_manager.get_logs()  # Grab everything from the LoggingManager
            }
            log_folder = "logs_and_log_tools"
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            log_filename = (f"{log_folder}/{self.experiment_name}_"
                            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")

            with open(log_filename, 'w') as f:
                json.dump(final_log, f, indent=4)

            # Also save the GA_Logger events if it exists
            if self.logger:
                self.logger.save()
