"""
test_population_manager.py

Unit tests for the PopulationManager: initialization, fitness evaluation,
and next-generation creation (elitism, crossover, mutation integration).
"""

import random
import unittest

from src.M_E_GA import M_E_GA_Base


class TestPopulationManager(unittest.TestCase):
    """
    Ensures that the PopulationManager creates and evolves populations correctly.
    """

    def setUp(self):
        """
        Create a small GA instance to access the population manager.
        """
        random.seed(2025)
        self.ga = M_E_GA_Base(
            genes=['0', '1'],
            fitness_function=lambda org, ga: sum(org),  # silly fitness summing integer codons
            population_size=4,
            max_individual_length=5,
            logging=False,
            experiment_name="TestPopulationManager"
        )
        self.pop_manager = self.ga.population_manager

    def test_initialize_population(self):
        """
        Check that the population is properly initialized.
        Only 'functional' genes should not exceed max_individual_length.
        Delimiters (Start/End) do not count against that length.
        """
        population = self.pop_manager.initialize_population()
        self.assertEqual(
            len(population),
            self.ga.population_size,
            "Population should match the configured size."
        )

        for individual in population:
            # Decode with format=True to strip Start/End
            functional_genes = self.ga.decode_organism(individual, format=True)
            self.assertLessEqual(
                len(functional_genes),
                self.ga.max_individual_length,
                "No individual should exceed the max functional gene length."
            )

    def test_evaluate_population_fitness(self):
        """
        Evaluate the population with a simple sum-based fitness function.
        """
        population = [
            [0, 1, 1],
            [1, 1, 1, 1],
            [0],
            [1, 0, 1]
        ]
        fitness_scores = self.pop_manager.evaluate_population_fitness(population)
        # We used sum of codons as fitness. Check correctness
        self.assertEqual(fitness_scores, [2, 4, 0, 2], "Fitness should be the sum of codons for each organism.")

    def test_select_and_generate_new_population(self):
        """
        Test we get the correct size new population, with elitism and random combos.
        """

        def custom_fitness(org, ga):
            # Let's pretend: we want more '1's = higher fitness
            return sum(org)

        self.ga.fitness_function = custom_fitness
        population = self.pop_manager.initialize_population()
        fitness_scores = self.pop_manager.evaluate_population_fitness(population)
        new_pop = self.pop_manager.select_and_generate_new_population(population, fitness_scores, 0)
        self.assertEqual(len(new_pop), self.ga.population_size,
                         "New population must remain consistent with population size.")

        # We won't attempt to decode or check specifics here, just confirm no crash or size mismatch.


if __name__ == '__main__':
    unittest.main()
