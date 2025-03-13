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
            logging=False,  # Turn on if you want logs
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
        for gene in ['A', 'B', 'C']:
            self.assertIn(gene, decoded,
                          f"Decoded organism should contain gene '{gene}'.")

    def test_run_algorithm_small(self):
        """
        Do a short run of the GA with a dummy fitness function.
        Confirm no errors occur, and that we end up with a final population/logs.
        """
        self.ga.run_algorithm()

        self.assertEqual(len(self.ga.population), self.ga.population_size,
                         "Final population size should remain consistent with population_size.")
        self.assertEqual(len(self.ga.fitness_scores), self.ga.population_size,
                         "We should have a fitness score for each individual in the final population.")

    def test_decode_organism_format_true(self):
        """
        Check that decode_organism with format=True strips out 'Start' and 'End'.
        """
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']
        organism = [start_codon,
                    self.ga.encoding_manager.reverse_encodings['A'],
                    self.ga.encoding_manager.reverse_encodings['B'],
                    end_codon]

        decoded_unformatted = self.ga.decode_organism(organism, format=False)
        self.assertIn('Start', decoded_unformatted, "Decoded organism (format=False) should keep 'Start' delimiter.")
        self.assertIn('End', decoded_unformatted, "Decoded organism (format=False) should keep 'End' delimiter.")

        decoded_formatted = self.ga.decode_organism(organism, format=True)
        self.assertNotIn('Start', decoded_formatted, "When format=True, 'Start' should be removed from decoded output.")
        self.assertNotIn('End', decoded_formatted, "When format=True, 'End' should be removed from decoded output.")
        self.assertIn('A', decoded_formatted, "Gene 'A' should remain in the formatted decode.")
        self.assertIn('B', decoded_formatted, "Gene 'B' should remain in the formatted decode.")

    def test_meta_gene_deletion_issue7(self):
        """
        Stress test to ensure that meta-gene deletion doesn't leave 'Unknown' references.
        We configure a GA with frequent meta-gene usage and strict decoding to see if
        any unknown references appear.
        """
        # Setup a new GA instance with heavier meta-gene usage
        ga = M_E_GA_Base(
            genes=['G1', 'G2', 'G3', 'G4', 'G5'],
            fitness_function=dummy_fitness_function,
            # Make meta-gene capturing more likely:
            metagene_mutation_prob=0.15,
            open_mutation_prob=0.10,
            delimiter_insert_prob=0.10,
            mutation_prob=0.05,
            delimited_mutation_prob=0.05,
            population_size=2000,
            num_parents=6,
            max_generations=500,
            strict_decode=True,  # <-- We want unknown references to throw exceptions
            logging=False
        )
        ga.initialize_population()

        # If the bug triggers "Unknown" references, an exception is raised inside run_algorithm
        try:
            ga.run_algorithm()
        except ValueError as e:
            self.fail(f"Meta-gene deletion flow caused 'Unknown' reference: {e}")

        # If we get here, we had no unknown references, so presumably the bug didn't occur or it's fixed.
        self.assertTrue(True, "No 'Unknown' references found during the meta-gene deletion stress test.")


if __name__ == '__main__':
    unittest.main()
