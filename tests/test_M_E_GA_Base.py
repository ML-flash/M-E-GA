import random
import unittest

from src.M_E_GA import EncodingManager, M_E_GA_Base


class DummyFitnessEvaluator:
    """
    A stub fitness evaluator class that returns random scores
    or zero. This can be expanded to replicate a specific scenario.
    """

    def evaluate(self, population, ga_instance):
        # Return a list of random fitness scores, one per individual
        return [random.uniform(0, 1) for _ in population]


def dummy_fitness_function(organism, ga_instance):
    """
    A simple fitness function stub that returns a random value.
    (Or you can interpret 'organism' in some way.)
    """
    return random.uniform(0, 1)


class TestM_E_GA_Base(unittest.TestCase):
    def setUp(self):
        """
        Set up a M_E_GA_Base instance with a small gene pool,
        plus a dummy fitness function and evaluator.
        """
        self.base_genes = ['A', 'B', 'C', 'D']
        self.ga = M_E_GA_Base(
            genes=self.base_genes,
            fitness_function=dummy_fitness_function,
            fitness_evaluator=DummyFitnessEvaluator(),
            logging=False,  # Turn on if you want to see JSON logs, e.g. True
            experiment_name="UnitTest_GA_Base",
            population_size=6,
            num_parents=2,
            max_generations=3,
            seed=42  # Ensures reproducibility for random operations
        )

    def test_initialization(self):
        """Check that M_E_GA_Base is initialized with the right default values."""
        self.assertIsInstance(self.ga.encoding_manager, EncodingManager,
                              "M_E_GA_Base should have an EncodingManager instance.")
        self.assertEqual(self.ga.population_size, 6,
                         "Population size should match the initialization parameter.")
        self.assertEqual(self.ga.num_parents, 2,
                         "Number of parents should match the initialization parameter.")
        self.assertEqual(self.ga.max_generations, 3,
                         "Max generations should match the initialization parameter.")
        self.assertEqual(self.ga.genes, self.base_genes,
                         "Gene set should match the ones provided in initialization.")

    def test_initialize_population(self):
        """Test that initializing the population creates the right number of organisms."""
        population = self.ga.initialize_population()
        self.assertEqual(len(population), self.ga.population_size,
                         "Population size should match the configured population_size.")
        # Check that each organism is a list (or tuple) of hash keys
        for organism in population:
            self.assertIsInstance(organism, list,
                                  "Each individual in the population should be a list of codons/hash keys.")

    def test_encode_decode_string(self):
        """Test encoding and decoding a simple string of genes."""
        test_string = ['A', 'B', 'C']
        encoded = self.ga.encode_string(test_string)
        self.assertIsInstance(encoded, list,
                              "Encoded result should be a list of hash keys.")

        # Now decode it back
        decoded = self.ga.decode_organism(encoded, format=False)
        # The decode might include 'Start'/'End' if used—depends on your manager setup
        # For a simple test, let's just confirm that 'A', 'B', 'C' appear in order.
        # The presence of optional delimiters won't break the test, we only assert containment.
        for gene in ['A', 'B', 'C']:
            self.assertIn(gene, decoded,
                          f"Decoded organism should contain gene '{gene}'.")

    def test_run_algorithm_small(self):
        """
        Do a short run of the GA with a dummy fitness function.
        Confirm no errors occur, and that we end up with a final population/logs.
        """
        # Run the GA (this calls initialize_population and goes through max_generations)
        self.ga.run_algorithm()

        # After run_algorithm(), we expect:
        #  1) self.ga.population is the final population
        #  2) self.ga.fitness_scores is the last generation's fitness
        self.assertEqual(len(self.ga.population), self.ga.population_size,
                         "Final population size should remain consistent with population_size.")
        self.assertEqual(len(self.ga.fitness_scores), self.ga.population_size,
                         "We should have a fitness score for each individual in the final population.")

        # If logging was enabled, check that logs are present
        if self.ga.logging:
            self.assertTrue(len(self.ga.logs) > 0,
                            "If logging is enabled, there should be logs for each generation.")

    def test_decode_organism_format_true(self):
        """
        Check that decode_organism with format=True strips out 'Start' and 'End'.
        This depends on how your M_E_Engine handles them.
        """
        # Make a small organism with potential Start/End included
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']
        organism = [start_codon,
                    self.ga.encoding_manager.reverse_encodings['A'],
                    self.ga.encoding_manager.reverse_encodings['B'],
                    end_codon]

        decoded_unformatted = self.ga.decode_organism(organism, format=False)
        # Should include 'Start', 'A', 'B', 'End'
        self.assertIn('Start', decoded_unformatted, "Decoded organism (format=False) should keep 'Start' delimiter.")
        self.assertIn('End', decoded_unformatted, "Decoded organism (format=False) should keep 'End' delimiter.")

        decoded_formatted = self.ga.decode_organism(organism, format=True)
        # Should remove 'Start' and 'End'
        self.assertNotIn('Start', decoded_formatted, "When format=True, 'Start' should be removed from decoded output.")
        self.assertNotIn('End', decoded_formatted, "When format=True, 'End' should be removed from decoded output.")
        # 'A' and 'B' should remain
        self.assertIn('A', decoded_formatted, "Gene 'A' should remain in the formatted decode.")
        self.assertIn('B', decoded_formatted, "Gene 'B' should remain in the formatted decode.")


if __name__ == '__main__':
    unittest.main()
