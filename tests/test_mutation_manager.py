"""
test_mutation_manager.py

Unit tests for the MutationManager and the underlying mutation operations.
We avoid randomness in critical tests by directly calling subfunctions
(e.g., capture, open) so the results are deterministic.
"""

import random
import unittest

from src.M_E_GA import M_E_GA_Base
from src.M_E_GA.engine.mutation.metagene_mutations import perform_open, perform_capture


class TestMutationManager(unittest.TestCase):
    """
    Tests the MutationManager's ability to apply insertion, deletion,
    point mutations, capture, open, etc.
    """

    def setUp(self):
        """
        Set up a minimal M_E_GA_Base instance with a small gene pool to keep tests stable.
        Also retrieve its MutationManager for direct testing.
        """
        random.seed(12345)  # for deterministic test outcomes

        self.ga = M_E_GA_Base(
            genes=['X', 'Y', 'Z'],
            fitness_function=lambda org, ga: 0.0,
            population_size=2,
            max_individual_length=5,
            logging=False,
            experiment_name="TestMutationManager"
        )
        # Force population init so we have something to mutate
        self.population = self.ga.initialize_population()
        self.mutation_manager = self.ga.mutation_manager

    def test_mutate_organism_insertion(self):
        """
        Test that insertion can occur, increasing organism length by one (unless random says otherwise).
        We won't force it, just ensure it doesn't crash.
        """
        organism = self.population[0][:]
        initial_len = len(organism)
        mutated = self.mutation_manager.mutate_organism(organism, generation=0)
        self.assertTrue(len(mutated) >= 1, "Mutated organism should have at least 1 gene.")
        # We won't check strict length difference, because it's random. Just ensuring no errors.

    def test_mutate_organism_point_mutation(self):
        """
        We artificially set high mutation probabilities so we know point mutation is likely,
        but not guaranteed. We'll just ensure no crash.
        """
        self.ga.mutation_prob = 1.0
        self.ga.delimiter_insert_prob = 0
        self.ga.open_mutation_prob = 0
        self.ga.metagene_mutation_prob = 0

        organism = self.population[0][:]
        mutated = self.mutation_manager.mutate_organism(organism, generation=0)
        self.assertGreater(len(mutated), 0, "Point mutation should keep length about the same.")
        # No crash = success for now.

    def test_capture_and_open_metagene(self):
        """
        Test capturing a delimited segment and then reopening it *deterministically* by
        calling the subfunctions directly instead of relying on random mutation selection.

        Steps:
          1. Build an organism [Start, X, Y, End].
          2. Call perform_capture() at the location of Start to compress X,Y into a metagene.
          3. Verify we get a single metagene codon in the organism.
          4. Call perform_open() on that codon to decompress it with delimiters again.
          5. Verify we see [Start, X, Y, End] in the final result.
        """
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']
        gene_x = self.ga.encoding_manager.reverse_encodings['X']
        gene_y = self.ga.encoding_manager.reverse_encodings['Y']

        # Step 1: Build an organism
        organism = [start_codon, gene_x, gene_y, end_codon]
        self.assertEqual(len(organism), 4, "We should start with 4 codons.")

        # Step 2: Capture the segment using the subfunction (rather than random mutate)
        # We'll assume the segment is from index=0 (Start) to index=3 (End).
        new_org, new_index, mutation_log = perform_capture(
            organism=organism,
            index=0,
            generation=1,
            manager=self.mutation_manager
        )

        # Step 3: We expect new_org to be a single-element list if capture succeeded:
        # i.e. [some_metagene_codon]
        self.assertEqual(len(new_org), 1, f"After capture, organism should shrink to 1 codon. Got {new_org}.")
        captured_codon = new_org[0]
        self.assertIn(captured_codon, self.ga.encoding_manager.meta_genes,
                      "Captured codon should be recognized as a metagene.")

        # Step 4: Now forcibly open that metagene with delimiters:
        fully_opened, final_idx, open_log = perform_open(
            organism=new_org,
            index=0,
            generation=2,
            manager=self.mutation_manager,
            no_delimit=False  # we want Start/End back
        )

        # Step 5: Verify the result is [Start, X, Y, End]
        self.assertEqual(len(fully_opened), 4, "Reopened organism should have 4 codons (Start, X, Y, End).")
        self.assertEqual(fully_opened[0], start_codon, "First codon should be Start.")
        self.assertEqual(fully_opened[1], gene_x, "Second codon should be X.")
        self.assertEqual(fully_opened[2], gene_y, "Third codon should be Y.")
        self.assertEqual(fully_opened[3], end_codon, "Fourth codon should be End.")

    def test_repair_unmatched_delimiters(self):
        """
        Test that unmatched delimiters get removed by repair().
        Example: [Start, Start, X, End] => one Start is unmatched and should be removed.
        """
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']
        organism = [start_codon, start_codon, self.ga.encoding_manager.reverse_encodings['X'], end_codon]

        repaired = self.mutation_manager.repair(organism)
        start_count = sum(1 for c in repaired if c == start_codon)
        end_count = sum(1 for c in repaired if c == end_codon)
        self.assertEqual(start_count, end_count,
                         "After repair, we should have equal Start and End codons (all matched).")


if __name__ == '__main__':
    unittest.main()
