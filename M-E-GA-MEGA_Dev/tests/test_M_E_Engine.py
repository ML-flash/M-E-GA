import random
import unittest

from src.M_E_GA import EncodingManager


class TestEncodingManager(unittest.TestCase):
    """Basic unit tests for the EncodingManager."""

    def setUp(self):
        """Initialize an EncodingManager before each test."""
        self.manager = EncodingManager()

    def test_initialization(self):
        """Test that the EncodingManager initializes correctly with default delimiters."""
        self.assertIn('Start', self.manager.reverse_encodings)
        self.assertIn('End', self.manager.reverse_encodings)
        self.assertEqual(self.manager.gene_counter, 3,
                         "Gene counter should start at 3 (since 1 and 2 are used by 'Start' and 'End').")

    def test_add_and_encode_genes(self):
        """Test adding genes and encoding them."""
        genes = ['A', 'B', 'C']
        encoded_keys = []
        for gene in genes:
            hash_key = self.manager.add_gene(gene)
            encoded_keys.append(hash_key)

        for gene, hash_key in zip(genes, encoded_keys):
            self.assertEqual(self.manager.reverse_encodings[gene], hash_key,
                             f"Gene '{gene}' should map to hash key {hash_key}.")

        encoded = self.manager.encode(genes)
        self.assertEqual(encoded, encoded_keys,
                         "Encoded sequence should match the expected hash keys.")

    def test_capture_and_open_metagene(self):
        """Test capturing a segment and opening the captured metagene with delimiters."""
        self.manager.add_gene('A')
        self.manager.add_gene('B')
        segment = self.manager.encode(['A', 'B'])
        captured_codon = self.manager.capture_metagene(segment)

        self.assertIn(captured_codon, self.manager.meta_genes,
                      "Captured codon should be in meta_genes.")

        opened_segment = self.manager.open_metagene(captured_codon)
        decoded_segment = self.manager.decode(tuple(opened_segment))

        expected = ['Start', 'A', 'B', 'End']
        self.assertEqual(decoded_segment, expected,
                         "Opened metagene should match the original segment with 'Start'/'End' delimiters.")

    def test_decode_unknown_hash_key(self):
        """Test decoding an unknown hash key returns 'Unknown'."""
        unknown_hash_key = 99999  # A hash key that doesn't exist
        decoded = self.manager.decode((unknown_hash_key,))
        self.assertIn('Unknown', decoded,
                      "Unknown hash key should decode to 'Unknown' at least once.")

    def test_duplicate_segment_capture(self):
        """Test that capturing the same segment multiple times reuses the same hash key."""
        self.manager.add_gene('X')
        self.manager.add_gene('Y')
        segment = self.manager.encode(['X', 'Y'])

        captured_codon_1 = self.manager.capture_metagene(segment)
        captured_codon_2 = self.manager.capture_metagene(segment)

        self.assertEqual(captured_codon_1, captured_codon_2,
                         "Duplicate segments should reuse the same metagene hash key.")

    def test_lru_cache_eviction(self):
        """Test that the LRU cache evicts the least recently used metagene when full."""
        # Fill up the LRU cache
        for i in range(self.manager.lru_cache_size + 1):
            gene = f"Gene_{i}"
            self.manager.add_gene(gene)
            encoded = self.manager.encode([gene])
            self.manager.capture_metagene(encoded)

        # The *oldest* captured metagene should have been evicted
        # We used '1' for 'Start', '2' for 'End', so let's check the hash key for integer 1
        oldest_codon = self.manager.generate_hash_key(1)

        self.assertNotIn(oldest_codon, self.manager.metagene_usage,
                         "Oldest metagene should have been evicted from the LRU cache.")


class TestRobustMetageneRecycling(unittest.TestCase):
    """
    A more advanced suite that creates a random web of metagenes,
    deletes some, and reuses freed codons. We ensure no 'Unknown'
    tokens appear in the decoding of 'active' metagenes.
    """

    def setUp(self):
        """Initialize an EncodingManager before each test."""
        self.manager = EncodingManager()
        self.manager.debug = False  # Toggle True for verbose debugging

        # Create a small pool of base genes
        base_genes = ['A', 'B', 'C', 'D', 'E', 'F']
        for g in base_genes:
            self.manager.add_gene(g)

    def build_random_segment(self, existing_metagenes, max_segment_size=5):
        """
        Build a random segment (list of items) referencing either base genes or existing metagenes.
        """
        segment_size = random.randint(1, max_segment_size)
        segment = []
        all_possible = list(self.manager.reverse_encodings.keys())  # base genes
        # also allow references to existing metagenes:
        all_possible.extend(existing_metagenes)

        for _ in range(segment_size):
            # 50/50 chance to choose either a base gene or an existing metagene reference
            choice = random.choice(all_possible)
            segment.append(choice)

        # Convert them to hash keys via encode()
        return self.manager.encode(segment)

    def test_robust_recycling(self):
        """
        Steps:
          1. Randomly create a bunch of metagenes referencing base genes or each other.
          2. Randomly delete some existing metagenes. This should inline references and produce no 'Unknown'.
          3. Reuse freed codons to build new metagenes; check they decode properly.
          4. Repeat multiple times to stress-test the manager.
        """
        existing_metagenes = []
        iterations = 20  # Increase for a more thorough stress-test

        for step in range(iterations):
            action = random.choice(["create", "delete", "reuse", "noop"])

            # --- CREATE ---
            if action == "create" or (not existing_metagenes):
                # Build a random segment
                segment = self.build_random_segment(existing_metagenes, max_segment_size=3)
                # Capture it
                new_mg = self.manager.capture_metagene(segment)
                existing_metagenes.append(new_mg)

            # --- DELETE ---
            elif action == "delete" and existing_metagenes:
                doomed = random.choice(existing_metagenes)
                self.manager.deletion_basket[doomed] = 2
                # Trigger the manager to process the deletion
                self.manager.start_new_generation()

                # If truly removed from meta_genes, drop from our local tracking
                if doomed not in self.manager.meta_genes:
                    existing_metagenes.remove(doomed)

            # --- REUSE ---
            elif action == "reuse" and self.manager.unused_encodings:
                # Reuse a freed codon by capturing another new segment
                segment = self.build_random_segment(existing_metagenes, max_segment_size=2)
                reused = self.manager.capture_metagene(segment)
                existing_metagenes.append(reused)

            # --- NOOP ---
            else:
                pass

            # Validate: every "active" metagene must decode with no 'Unknown'
            for mg in existing_metagenes:
                if mg in self.manager.meta_genes:
                    decoded = self.manager.decode((mg,))
                    self.assertNotIn(
                        'Unknown', decoded,
                        f"Metagene {mg} decoded as 'Unknown' but was supposed to be valid: {decoded}"
                    )

        # After final iteration, do one more generation pass
        self.manager.start_new_generation()

        # Final check: decode all still-active metagenes
        for mg in existing_metagenes:
            if mg in self.manager.meta_genes:
                decoded = self.manager.decode((mg,))
                self.assertNotIn(
                    'Unknown', decoded,
                    f"Final check: Metagene {mg} unexpectedly contains 'Unknown'."
                )


if __name__ == '__main__':
    unittest.main()
