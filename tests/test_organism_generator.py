"""
test_organism_generator.py

Unit tests for the OrganismGenerator class, ensuring random generation
correctly uses functional_length, spacing, and optional delimiters.
"""

import random
import unittest

from src.M_E_GA import M_E_GA_Base


class TestOrganismGenerator(unittest.TestCase):
    """
    Validate the OrganismGenerator's random creation logic.
    """

    def setUp(self):
        random.seed(777)
        self.ga = M_E_GA_Base(
            genes=['GeneA', 'GeneB', 'GeneC'],
            fitness_function=lambda org, ga: 0,
            logging=False,
            population_size=1,
            experiment_name="TestOrganismGenerator"
        )
        self.generator = self.ga.encoding_manager.organism_generator

    def test_generate_random_organism_basic(self):
        """
        Test generating a random organism without delimiters.
        """
        org = self.generator.generate_random_organism(
            functional_length=5,
            include_specials=False,
            special_spacing=2,
            probability=1.0
        )
        # Expect a length = 5 of random base genes
        self.assertEqual(len(org), 5, "Functional length should match the requested length.")

    def test_generate_random_organism_with_delimiters(self):
        """
        Test generating an organism with delimiters possibly inserted.
        """
        org = self.generator.generate_random_organism(
            functional_length=5,
            include_specials=True,
            special_spacing=2,
            probability=0.5
        )
        # We can't be certain how many Start/End will appear. Just ensure no crash and it's at least functional_length
        self.assertGreaterEqual(len(org), 5, "Generated organism with delimiters should have at least 5 codons.")


if __name__ == '__main__':
    unittest.main()
