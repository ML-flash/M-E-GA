import unittest
from M_E_GA.M_E_Engine import EncodingManager


class TestEncodingManager(unittest.TestCase):
    def setUp(self):
        """Initialize an EncodingManager before each test."""
        self.manager = EncodingManager()

    def test_add_and_encode_genes(self):
        """Test adding genes and encoding them."""
        genes = ['A', 'B', 'C']
        for gene in genes:
            self.manager.add_gene(gene, verbose=True)

        for gene in genes:
            encoded = tuple(self.manager.encode([gene, 'End'], verbose=True))
            decoded = self.manager.decode(encoded, verbose=True)
            expected_decoded_str = gene + ' End'
            decoded_str = ' '.join(decoded)
            self.assertEqual(decoded_str, expected_decoded_str,
                             f"Encoded sequence should decode back to '{expected_decoded_str}'.")

    def test_decode_unknown_hash_key(self):
        """Test decoding of an unknown hash key."""
        self.manager.add_gene('UniqueGene', verbose=False)
        unknown_hash_key = self.manager.gene_counter + 1
        encoded = tuple([unknown_hash_key])
        decoded = self.manager.decode(encoded, verbose=True)
        self.assertIn('Unknown', decoded, "Unknown hash key should decode to 'Unknown'.")


class TestEncodingManagerCapturedSegments(unittest.TestCase):
    def setUp(self):
        """Initialize an EncodingManager before each test."""
        self.manager = EncodingManager()

    def test_capture_and_decode(self):
        """Test capturing and decoding segments."""
        self.manager.add_gene('A')
        self.manager.add_gene('B')
        encoded_segment = self.manager.encode(['A', 'B'], verbose=True)
        captured_codon = self.manager.capture_segment(encoded_segment, verbose=True)
        decoded_sequence = self.manager.decode(tuple([captured_codon]), verbose=True)
        decoded_str = ' '.join(decoded_sequence)
        self.assertEqual(decoded_str, 'A B', "The decoded sequence should match the original segment.")

    def test_explicit_nested_capture_and_decoding(self):
        """Test capturing and decoding nested segments."""
        genes = ['1', '2', '3', '4', '5']
        for gene in genes:
            self.manager.add_gene(gene, verbose=True)
        encoded_segment = []
        for gene in genes:
            encoded_gene = self.manager.encode([gene], verbose=True)
            encoded_segment.extend(encoded_gene)
            captured_codon = self.manager.capture_segment(encoded_segment, verbose=True)
            decoded_sequence = self.manager.decode(tuple([captured_codon]), verbose=True)
            decoded_str = ' '.join(decoded_sequence)
            expected_decoded_str = ' '.join(genes[:len(decoded_sequence)])
            self.assertEqual(decoded_str, expected_decoded_str,
                             f"Decoded sequence should match {expected_decoded_str}")

    def test_duplicate_segment_capture(self):
        """Test that duplicate segments use the same hash key."""
        self.manager.add_gene('X')
        self.manager.add_gene('Y')
        encoded_segment_1 = self.manager.encode(['X', 'Y'], verbose=True)
        captured_codon_1 = self.manager.capture_segment(encoded_segment_1, verbose=True)
        encoded_segment_2 = self.manager.encode(['X', 'Y'], verbose=True)
        captured_codon_2 = self.manager.capture_segment(encoded_segment_2, verbose=True)
        self.assertEqual(captured_codon_1, captured_codon_2, "Duplicate segments should reuse the same hash key.")
        decoded_sequence_1 = self.manager.decode(tuple([captured_codon_1]), verbose=True)
        decoded_sequence_2 = self.manager.decode(tuple([captured_codon_2]), verbose=True)
        decoded_str_1 = ' '.join(decoded_sequence_1)
        decoded_str_2 = ' '.join(decoded_sequence_2)
        self.assertEqual(decoded_str_1, decoded_str_2, "Decoded sequences from duplicate captures should be identical.")


if __name__ == '__main__':
    unittest.main()
