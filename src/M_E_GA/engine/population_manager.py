"""
population_manager.py

Handles population-level operations: initialization, fitness evaluation,
and generating the next generation via selection, elitism, crossover, and mutation.
"""

import random


class PopulationManager:
    """
    PopulationManager is responsible for:
      - Initializing a population
      - Evaluating fitness (in conjunction with a fitness function or evaluator)
      - Selecting parents, applying crossover/mutation, and building the new generation.
    """

    def __init__(self, ga_instance):
        """
        Initialize the PopulationManager with a reference to the GA instance.

        :param ga_instance: The main M_E_GA_Base instance that orchestrates everything.
        """
        self.ga = ga_instance

    def initialize_population(self):
        """
        Generate an initial population of organisms.

        :return: A list of organism encodings representing the initial population.
        """
        population = []
        for _ in range(int(self.ga.population_size)):
            individual_length = random.randint(2, self.ga.max_individual_length)
            organism = self.ga.encoding_manager.generate_random_organism(
                functional_length=individual_length,
                include_specials=self.ga.delimiters,
                probability=0.10,
                verbose=False
            )
            population.append(organism)
        return population

    def evaluate_population_fitness(self, population):
        """
        Evaluate the fitness of the entire population.

        This method optionally calls a pre-evaluation callback, evaluates the population
        using the fitness evaluator, and then calls a post-evaluation callback.

        :param population: The list of organisms to evaluate.
        :return: A list of fitness scores for the population.
        """
        if self.ga.before_fitness_evaluation:
            self.ga.before_fitness_evaluation(self.ga)

        # If we have a dedicated fitness evaluator object, let that handle evaluations:
        if self.ga.fitness_evaluator is not None:
            fitness_scores = self.ga.fitness_evaluator.evaluate(population, self.ga)
        else:
            # Otherwise, fallback to the fitness_function callable
            fitness_scores = [self.ga.fitness_function(ind, self.ga) for ind in population]

        if self.ga.after_population_selection:
            self.ga.after_population_selection(self.ga)

        return fitness_scores

    def select_and_generate_new_population(self, population, fitness_scores, generation):
        """
        Select individuals and generate a new population through reproduction.

        Implements:
          - Elitism
          - Parent selection
          - Crossover
          - Mutation

        :param population: Current population of organisms.
        :param fitness_scores: List of corresponding fitness scores.
        :param generation: Current generation number (used for logging).
        :return: The new population of organisms (list of encodings).
        """
        # Sort population by fitness
        sorted_population = sorted(zip(population, fitness_scores),
                                   key=lambda x: x[1], reverse=True)

        # Get elites
        num_elites = int(self.ga.elitism_ratio * self.ga.population_size)
        elites = [individual for (individual, _) in sorted_population[:num_elites]]

        new_population = elites[:]

        # Prepare the pool of parents
        selected_parents = [individual for (individual, _) in sorted_population[:self.ga.num_parents]]
        shift = 0

        # Keep filling the new population until it hits target size
        while len(new_population) < self.ga.population_size:
            for i in range(0, len(selected_parents) - 1, 2):
                # We do a small shift each iteration to vary pairings
                parent1_index = (i + shift) % len(selected_parents)
                parent2_index = (i + 1 + shift) % len(selected_parents)
                parent1 = selected_parents[parent1_index]
                parent2 = selected_parents[parent2_index]

                # If fully delimited, skip crossover+mutation
                if self.ga.crossover_manager.is_fully_delimited(parent1) or \
                        self.ga.crossover_manager.is_fully_delimited(parent2):
                    new_population.extend([parent1, parent2][:self.ga.population_size - len(new_population)])
                    continue

                # Possibly apply crossover
                if random.random() < self.ga.crossover_prob:
                    non_del_indices = self.ga.crossover_manager.get_non_delimiter_indices(parent1, parent2)
                    offspring1, offspring2 = self.ga.crossover_manager.crossover(
                        parent1, parent2, non_del_indices, generation
                    )
                else:
                    offspring1, offspring2 = parent1[:], parent2[:]

                # Updated references to the logging manager
                self.ga.logging_manager.log_new_organism(offspring1)
                self.ga.logging_manager.log_new_organism(offspring2)

                # Mutate
                offspring1 = self.ga.mutation_manager.mutate_organism(offspring1, generation)
                offspring2 = self.ga.mutation_manager.mutate_organism(offspring2, generation)

                # Add them
                new_population.extend(
                    [offspring1, offspring2][:self.ga.population_size - len(new_population)]
                )

            shift += 1

        return new_population
