# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:48:15 2024

@author: Matt Andrews
"""



import random
class EncodingManager:
    def __init__(self):
        self.encodings = {0b01: 'Start', 0b10: 'End'}
        self.reverse_encodings = {'Start': 0b01, 'End': 0b10}
        self.codon_size = 2
        self.next_codon_value = 0b11  # Start from a value that doesn't conflict with 'Start' and 'End'
        self.captured_segments = {}

    def add_gene(self, gene, verbose=False):
        if gene in self.reverse_encodings:
            if verbose:
                print(f"Gene '{gene}' is already added.")
            return

        while self.next_codon_value in self.encodings or self.next_codon_value == 0:
            self.next_codon_value += 1
            if self.next_codon_value >= (1 << self.codon_size):
                self.expand_codon_size(verbose=verbose)

        self.encodings[self.next_codon_value] = gene
        self.reverse_encodings[gene] = self.next_codon_value
        if verbose:
            print(f"Added gene '{gene}' with codon {bin(self.next_codon_value)}.")
        self.next_codon_value += 1
        
        
        
    def integrate_uploaded_encodings(self, uploaded_encodings, base_genes):
        if isinstance(uploaded_encodings, str):
            uploaded_encodings = {int(k): v for k, v in (item.split(':') for item in uploaded_encodings.split(','))}
    
        default_genes = {'Start': 0b01, 'End': 0b10}
        for gene, codon in default_genes.items():
            if gene in uploaded_encodings and uploaded_encodings[gene] != codon:
                raise ValueError(f"Default gene '{gene}' in uploaded encodings does not match the expected codon.")
    
        for key, value in uploaded_encodings.items():
            if isinstance(value, str):  # Base genes
                if value in base_genes or value in default_genes:
                    if value not in self.reverse_encodings:
                        self.encodings[key] = value
                        self.reverse_encodings[value] = key
                else:
                    raise ValueError(f"Base gene '{value}' in uploaded encodings does not match the expected base genes.")
    
        # Adjust how captured segments are integrated
        for key, value in uploaded_encodings.items():
            if isinstance(value, tuple):  # Captured segments
                # Ensure the key is an integer representing the captured codon
                # and the value is a tuple representing the original segment
                self.captured_segments[value] = key
                # Reverse the mapping for decoding: map integer codon to tuple
                self.encodings[key] = value
    
        # Update next_codon_value to avoid conflicts
        max_key = max(self.encodings.keys(), default=0, key=lambda k: k if isinstance(k, int) else 0)
        self.next_codon_value = max(self.next_codon_value, max_key + 1)
    
        # Adjust codon size if necessary
        while self.next_codon_value >= (1 << self.codon_size):
            self.expand_codon_size()





                
            
            

    def expand_codon_size(self, verbose=False):
        old_codon_size = self.codon_size
        self.codon_size += 1
        if verbose:
            print(f"Expanding codon size from {old_codon_size} to {self.codon_size} bits.")

    def encode(self, genetic_sequence, verbose=False):
        encoded_list = []
        genes = genetic_sequence.split()
    
        for gene in genes:
            codon = self.reverse_encodings.get(gene, 'Unknown')
            if codon == 'Unknown':
                if verbose:
                    print(f"Gene '{gene}' is not recognized.")
                continue
            encoded_list.append(bin(codon))
            if verbose:
                print(f"Encoding gene '{gene}' to codon {bin(codon)}.")
    
        return encoded_list




    def decode(self, encoded_list, verbose=False):
        decoded_sequence = []
        stack = encoded_list[:]  # Copy the encoded list to a stack for processing
    
        while stack:
            binary_codon = stack.pop(0)  # Pop the first item for decoding
            codon = int(binary_codon, 2) if isinstance(binary_codon, str) and binary_codon.startswith('0b') else binary_codon
    
            if codon in self.encodings:
                value = self.encodings[codon]
    
                if isinstance(value, tuple):  # Handling captured segments
                    if verbose:
                        print(f"Decompressing captured codon {binary_codon}")
                    # Push the contents of the captured segment to the start of the stack for decoding
                    stack = list(value) + stack
                else:
                    # Direct mapping of codon to gene, append the value to the decoded list
                    decoded_sequence.append(value)
                    if verbose:
                        print(f"Decoding codon {binary_codon} to '{value}'.")
    
            else:
                decoded_sequence.append("Unknown")
                if verbose:
                    print(f"Codon {binary_codon} is unknown.")
    
        return decoded_sequence  # Return a list of decoded elements





    
    

    def capture_segment(self, encoded_segment, verbose=False):
        
        # Encapsulate the binary encodings in a tuple
        captured_key = tuple(encoded_segment)
        
        # Expand codon size if the next value exceeds current capacity
        if self.next_codon_value >= (1 << self.codon_size):
            self.expand_codon_size(verbose=verbose)
    
        # Assign a new binary encoding to the captured segment
        self.captured_segments[captured_key] = self.next_codon_value
        self.encodings[self.next_codon_value] = captured_key
    
        if verbose:
            print(f"Capturing segment {captured_key} with new codon {bin(self.next_codon_value)}.")
            print(f"Current encodings: {self.encodings}")
            print(f"Current captured segments: {self.captured_segments}")
    
        # Prepare for the next codon value
        self.next_codon_value += 1
    
        return self.captured_segments[captured_key]
    
    def open_segment(self, binary_codon, no_delimit=False, verbose=False):
        decompressed_codons = []
    
        # Convert binary codon to integer for dictionary lookup
        codon_int = int(binary_codon, 2)
        if verbose:
            print('The passed-in binary is', binary_codon, '. The int of it is', codon_int)
        
        # Use .get() to safely access the dictionary and avoid KeyError
        encoded_item = self.encodings.get(codon_int)
        
        # Check if the encoded_item exists and is a tuple
        if encoded_item and isinstance(encoded_item, tuple):
            if verbose:
                print(f"Decompressing captured segment for codon {binary_codon}.")
    
            if not no_delimit:
                # Add start delimiter if no_delimit is False
                start_delimiter_codon = bin(self.reverse_encodings['Start'])
                decompressed_codons.append(start_delimiter_codon)
            
            # Iterate through the tuple and add each codon to the decompressed_codons list
            for gene_codon in encoded_item:
                decompressed_codons.append(gene_codon)  # gene_codon is already a binary string
            
            if not no_delimit:
                # Add end delimiter if no_delimit is False
                end_delimiter_codon = bin(self.reverse_encodings['End'])
                decompressed_codons.append(end_delimiter_codon)
        else:
            if verbose:
                print(f"Codon {binary_codon} is not a captured segment or is unknown, returning as is.")
            decompressed_codons.append(binary_codon)
    
        return decompressed_codons

    
    
    def generate_random_organism(self, functional_length=100, include_specials=True, special_spacing=10, probability=0.99, verbose=False):
        gene_pool = [gene for gene in self.reverse_encodings if gene not in ['Start', 'End']]
        organism_genes = [random.choice(gene_pool) for _ in range(functional_length)]
        special_gene_indices = set()
    
        if include_specials:
            for i in range(len(organism_genes)):
                if random.random() <= probability:
                    # Check spacing from the last special gene
                    if all(abs(i - idx) >= special_spacing for idx in special_gene_indices):
                        # Insert 'Start' and 'End', ensuring 'End' is within bounds
                        organism_genes.insert(i, 'Start')
                        end_index = min(i + special_spacing, len(organism_genes))
                        organism_genes.insert(end_index, 'End')
                        # Update indices to include new special genes
                        special_gene_indices.update([i, end_index])
    
        encoded_organism = self.encode(' '.join(organism_genes))
    
        if verbose:
            print("Generated Encoded Organism:", encoded_organism)
    
        return encoded_organism










import unittest

class TestEncodingManager(unittest.TestCase):
    def setUp(self):
        self.manager = EncodingManager()

    def test_add_and_encode_genes(self):
        genes = ['A', 'B', 'C']
        for gene in genes:
            self.manager.add_gene(gene, verbose=True)
        for gene in genes:
            encoded = self.manager.encode(gene + ' End', verbose=True)
            decoded = self.manager.decode(encoded, verbose=True)
            # Join the decoded list to form a string for comparison
            decoded_str = ' '.join(decoded)
            self.assertEqual(decoded_str, gene + ' End', f"Encoded sequence should decode back to '{gene} End'.")

    def test_decode_unknown_codon(self):
        unknown_codon = 0b1111  # Use a codon that is likely not in use
        decoded = self.manager.decode([bin(unknown_codon)], verbose=True)
        # Check if 'Unknown' is in the decoded list
        self.assertIn('Unknown', decoded, "Unknown codon should decode to 'Unknown'.")

class TestEncodingManagerCapturedSegments(unittest.TestCase):
    def setUp(self):
        self.manager = EncodingManager()

    def test_capture_and_decode(self):
        self.manager.add_gene('A')
        self.manager.add_gene('B')
        encoded_segment = self.manager.encode('A B', verbose=True)
        captured_codon = self.manager.capture_segment(encoded_segment, verbose=True)
        decoded_sequence = self.manager.decode([captured_codon], verbose=True)
        # Join the decoded list to form a string for comparison
        decoded_str = ' '.join(decoded_sequence)
        self.assertEqual(decoded_str, 'A B', "The decoded sequence should match the original segment.")

    def test_explicit_nested_capture_and_decoding(self):
        genes = ['1', '2', '3', '4', '5']
        for gene in genes:
            self.manager.add_gene(gene, verbose=True)
        encoded_segment = []
        for gene in genes:
            encoded_gene = self.manager.encode(gene, verbose=True)
            encoded_segment.extend(encoded_gene)
            captured_codon = self.manager.capture_segment(encoded_segment, verbose=True)
            decoded_sequence = self.manager.decode([captured_codon], verbose=True)
            # Join the decoded list to form a string for comparison
            decoded_str = ' '.join(decoded_sequence)
            self.assertEqual(decoded_str, ' '.join(genes[:len(encoded_segment)]), f"Decoded sequence should match {' '.join(genes[:len(encoded_segment)])}")

class TestEncodingManagerNestedCaptures(unittest.TestCase):
    def setUp(self):
        self.manager = EncodingManager()

    def test_nested_segment_capture_with_multiple_genes(self):
        genes = ['X', 'Y', 'Z', 'W']
        for gene in genes:
            self.manager.add_gene(gene, verbose=True)
        initial_encoded_segment = self.manager.encode('X Y', verbose=True)
        initial_capture_codon = self.manager.capture_segment(initial_encoded_segment, verbose=True)
        # Use the binary representation of the initial_capture_codon in the nested segment
        nested_encoded_segment = [bin(initial_capture_codon)] + self.manager.encode('Z W', verbose=True)
        nested_capture_codon = self.manager.capture_segment(nested_encoded_segment, verbose=True)
        decoded_nested_sequence = self.manager.decode([nested_capture_codon], verbose=True)
        # Join the decoded list to form a string for comparison
        decoded_str = ' '.join(decoded_nested_sequence)
        self.assertEqual(decoded_str, 'X Y Z W', "Nested decoded segment should match 'X Y Z W'")
        

class TestOpenSegment(unittest.TestCase):
    def setUp(self):
        self.encoding_manager = EncodingManager()
        # Add genes and capture segments as needed
        self.genes = ['A', 'B', 'C', 'D']
        for gene in self.genes:
            self.encoding_manager.add_gene(gene)
        # Encode genes to create segments
        self.single_segment = [self.encoding_manager.encode(gene)[0] for gene in self.genes[:3]]  # ['A', 'B', 'C']
        self.single_captured_codon = bin(self.encoding_manager.capture_segment(self.single_segment))
        # For nested capture, include the captured codon in a new segment with 'D'
        self.nested_segment = [self.single_captured_codon] + [self.encoding_manager.encode(self.genes[3])[0]]
        self.nested_captured_codon = bin(self.encoding_manager.capture_segment(self.nested_segment))

    def test_single_level_capture_and_open_with_delimiters(self):
        # Open with delimiters
        opened_segment = self.encoding_manager.open_segment(self.single_captured_codon, no_delimit=False)
        expected_segment = [bin(self.encoding_manager.reverse_encodings['Start'])] + self.single_segment + [bin(self.encoding_manager.reverse_encodings['End'])]
        self.assertEqual(opened_segment, expected_segment, "Opened segment with delimiters does not match expected.")

    def test_single_level_capture_and_open_without_delimiters(self):
        # Open without delimiters
        opened_segment = self.encoding_manager.open_segment(self.single_captured_codon, no_delimit=True)
        self.assertEqual(opened_segment, self.single_segment, "Opened segment without delimiters does not match expected.")

    def test_nested_capture_and_open(self):
        # Open nested capture with no_delimit=True should not decompress the inner capture
        opened_nested_segment = self.encoding_manager.open_segment(self.nested_captured_codon, no_delimit=True)
        expected_nested_segment = [self.single_captured_codon, self.encoding_manager.encode(self.genes[3])[0]]
        self.assertEqual(opened_nested_segment, expected_nested_segment, "Opened nested segment does not match expected.")



class TestEncodingIntegrationAndGeneSelection(unittest.TestCase):
    def setUp(self):
        # Initialize the initial EncodingManager with predefined encodings
        self.initial_manager = EncodingManager()
        self.genes = ['A', 'B', 'C', 'D']
        for gene in self.genes:
            self.initial_manager.add_gene(gene)
        # Encode genes to create segments
        self.segment = [self.initial_manager.encode(gene)[0] for gene in self.genes[:2]]  # Example: ['A', 'B']
        # Capture the segment to create a captured codon
        self.captured_codon = self.initial_manager.capture_segment(self.segment)
        # Define the base gene probability for the test
        self.base_gene_prob = 0.8  # Example probability value

    def select_gene(self, manager):
        if random.random() < self.base_gene_prob or not manager.captured_segments:
            base_gene = random.choice(self.genes)
            gene_key = manager.reverse_encodings[base_gene]
        else:
            captured_segment = random.choice(list(manager.captured_segments.keys()))
            gene_key = manager.captured_segments[captured_segment]
        gene_bin = bin(gene_key)
        return gene_bin

    def test_integration_and_gene_selection(self):
        # Reinitialize EncodingManager and integrate encodings, passing in base genes
        new_manager = EncodingManager()
        new_manager.integrate_uploaded_encodings(self.initial_manager.encodings, self.genes)

        # Testing Opening of Captured Segments
        print("\n--- Opening Captured Segments ---")
        for captured_segment_tuple in new_manager.captured_segments.keys():
            captured_codon_value = new_manager.captured_segments[captured_segment_tuple]
            binary_codon = bin(captured_codon_value)
            opened_segment = new_manager.open_segment(binary_codon, no_delimit=False, verbose=True)
            print(f"Opened Segment from {binary_codon}: {opened_segment}")

        # Test gene selection multiple times to cover both base genes and captured segments
        for _ in range(10):
            selected_gene_bin = self.select_gene(new_manager)
            selected_gene_key = int(selected_gene_bin, 2)
            self.assertTrue(
                selected_gene_key in new_manager.encodings or
                selected_gene_key in new_manager.captured_segments.values(),
                "Selected gene should be from either base genes or captured segments."
            )




#if __name__ == '__main__':
#    unittest.main(verbosity=2)