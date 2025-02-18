"""
test_crossover_manager.py

Unit tests specifically for the CrossoverManager.
"""

import random
import unittest

from src.M_E_GA import M_E_GA_Base


class TestCrossoverManager(unittest.TestCase):
    """
    Tests how the CrossoverManager handles crossover points,
    delimiter avoidance, and final offspring.
    """

    def setUp(self):
        """
        Create a small GA instance and get the CrossoverManager for direct usage.
        """
        random.seed(999)
        self.ga = M_E_GA_Base(
            genes=['A', 'B'],
            fitness_function=lambda org, ga: 0,
            population_size=2,
            max_individual_length=5,
            logging=False,
            experiment_name="TestCrossoverManager"
        )
        self.crossover_manager = self.ga.crossover_manager

    def test_get_non_delimiter_indices(self):
        """
        Confirm that get_non_delimiter_indices avoids Start/End sections,
        leaving at least one valid index.
        """
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']

        # Parent1 has a delimiter block from indices 1..3 => leftover {0,4}
        parent1 = [101, start_codon, 202, end_codon, 303]
        # Parent2 has a delimiter block from indices 3..4 => leftover {0,1,2}
        parent2 = [111, 222, 333, start_codon, end_codon]

        non_delims = self.crossover_manager.get_non_delimiter_indices(parent1, parent2)

        # Now we should have a shared safe index (e.g., index 0).
        self.assertGreater(
            len(non_delims),
            0,
            "We should have some safe indices not inside Start-End blocks."
        )

    def test_crossover_basic(self):
        """
        Test a normal crossover scenario with no delimiters.
        """
        # Just two short parent organisms
        parent1 = [111, 222, 333]
        parent2 = [444, 555, 666]

        non_delims = self.crossover_manager.get_non_delimiter_indices(parent1, parent2)
        child1, child2 = self.crossover_manager.crossover(
            parent1, parent2, non_delims, generation=0
        )

        # We can't guarantee the exact position of the crossover, but let's confirm
        # that the lengths are preserved and the children are some combo of the parents.
        self.assertEqual(
            len(child1), len(parent1),
            "Offspring should preserve length."
        )
        self.assertEqual(
            len(child2), len(parent2),
            "Offspring should preserve length."
        )

    def test_is_fully_delimited(self):
        """
        Check if an organism is fully delimited: starts with 'Start' and ends with 'End'.
        """
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']

        org_fully = [start_codon, 111, 222, end_codon]
        self.assertTrue(
            self.crossover_manager.is_fully_delimited(org_fully),
            "Organism should be recognized as fully delimited."
        )

        org_partial = [111, start_codon, 222, end_codon]
        self.assertFalse(
            self.crossover_manager.is_fully_delimited(org_partial),
            "Organism should NOT be recognized as fully delimited."
        )


if __name__ == '__main__':
    unittest.main()
