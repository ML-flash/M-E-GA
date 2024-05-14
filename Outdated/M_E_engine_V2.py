# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:48:15 2024

@author: Matt Andrews
"""

import random
import xxhash
import functools

class EncodingManager:
     """
    Manages the encoding and decoding of genes using hash keys.
    
    This class handles the addition of new genes, integrates uploaded encoding data,
    and performs encoding and decoding of gene sequences. It supports the manipulation
    of captured gene segments and facilitates complex operations on encoded organisms.
    
    Attributes:
        encodings (dict): A dictionary mapping hash keys to gene names.
        reverse_encodings (dict): A dictionary mapping gene names to hash keys.
        captured_segments (dict): A dictionary mapping captured segment tuples to their hash keys.
        gene_counter (int): A counter to generate unique identifiers for new genes.
    """
    def __init__(self):
        """
        Initializes the EncodingManager with default genes 'Start' and 'End'.
        """
        # Initialize with default genes 'Start' and 'End'
        self.encodings = {}
        self.reverse_encodings = {}
        self.captured_segments = {}
        self.gene_counter = 3  # Start the counter from 3 after 'Start' and 'End'
        
        # Add default delimiters with predefined unique IDs
        self.add_gene('Start', predefined_id=1)
        self.add_gene('End', predefined_id=2)

    def generate_hash_key(self, identifier):
        """
        Generates a hash key for a given identifier using xxhash's 32-bit version.
        
        Parameters:
            identifier (str): The identifier for which to generate a hash key.
        
        Returns:
            int: A hash key generated from the identifier.
        """
        # Use xxhash's 32-bit version to generate a shorter hash
        return xxhash.xxh32_intdigest(str(identifier))

    def add_gene(self, gene, verbose=False, predefined_id=None):
        """
        Adds a gene to the encoding manager with an optional predefined identifier.
        
        Parameters:
            gene (str): The gene to be added.
            verbose (bool): If True, prints detailed debug information.
            predefined_id (int, optional): A predefined identifier for the gene.
        """
        # Use predefined_id for default genes or increment gene_counter for new genes
        identifier = predefined_id if predefined_id is not None else self.gene_counter

        if gene in self.reverse_encodings:
            if verbose:
                print(f"Gene '{gene}' is already added.")
            return

        # Generate hash key based on the unique identifier
        hash_key = self.generate_hash_key(identifier)

        self.encodings[hash_key] = gene
        self.reverse_encodings[gene] = hash_key
        if verbose:
            print(f"Added gene '{gene}' with hash key {hash_key}.")

        # Increment the counter for the next gene, if not using predefined_id
        if predefined_id is None:
            self.gene_counter += 1
        
        
        
    def integrate_uploaded_encodings(self, uploaded_encodings, base_genes, verbose=False):
        """
        Integrates uploaded encodings into the current manager, ensuring that default and base genes are correctly mapped.
        
        Parameters:
            uploaded_encodings (dict or str): Encodings uploaded from an external source, can be a dictionary or a string that needs parsing.
            base_genes (list): A list of base genes expected to be present in the uploaded encodings.
            verbose (bool): If True, prints detailed debug information.
        """
        if verbose:
            print("Starting integration of uploaded encodings...")
    
        if isinstance(uploaded_encodings, str):
            uploaded_encodings = {int(k): v for k, v in (item.split(':') for item in uploaded_encodings.split(','))}
            if verbose:
                print("Uploaded encodings after parsing:", uploaded_encodings)
    
        # Identify the hash keys for default genes 'Start' and 'End' from the initial manager
        start_key = self.reverse_encodings.get('Start')
        end_key = self.reverse_encodings.get('End')
        if verbose:
            print(f"Default gene 'Start' hash key: {start_key}, 'End' hash key: {end_key}")
    
        # Integrate base and default genes along with captured segments
        for key, value in uploaded_encodings.items():
            if value in base_genes or key in [start_key, end_key]:
                if value not in self.reverse_encodings or key in [start_key, end_key]:
                    self.encodings[key] = value
                    self.reverse_encodings[value] = key
                    if verbose:
                        print(f"Integrated gene '{value}' with key '{key}'.")
                else:
                    if verbose:
                        print(f"Gene '{value}' already in reverse encodings.")
            elif isinstance(value, tuple):  # Handle captured segments
                self.captured_segments[value] = key
                self.encodings[key] = value
                if verbose:
                    print(f"Integrated captured segment '{value}' with key '{key}'.")
            else:
                if verbose:
                    print(f"Skipping gene '{value}' with key '{key}' as it does not match expected base genes or default genes.")
    
        # Update gene counter to avoid conflicts
        max_hash_key = max(self.encodings.keys(), default=0)
        self.gene_counter = max(self.gene_counter, max_hash_key + 1)
        if verbose:
            print("Final updated gene counter:", self.gene_counter)


   
            
            

    def encode(self, genes, verbose=False):
        """
        Encodes a list of gene names into their corresponding hash keys.
        
        Parameters:
            genes (list of str): A list of gene names to be encoded.
            verbose (bool): If True, prints detailed debug information.
        
        Returns:
            list of int: A list of hash keys corresponding to the genes.
        """
        encoded_list = []
    
        for gene in genes:  # Directly iterate over each gene in the list
            # No conversion, direct retrieval
            hash_key = self.reverse_encodings.get(gene)
            if hash_key is None:
                if verbose:
                    # Print the gene as it is, without assuming it's a string or any other type
                    print(f"Gene '{gene}' is not recognized.")
                continue  # Skip unrecognized genes but continue processing
    
            encoded_list.append(hash_key)  # Add the hash key to the encoded list
    
            if verbose:
                # Print the gene as it is, directly
                print(f"Encoding gene '{gene}' to hash key {hash_key}.")
    
        return encoded_list  # Return the list of hash keys    

    @functools.lru_cache(maxsize=1000)
    def decode(self, encoded_tuple, verbose=False):
        """
        Decodes a tuple of hash keys back into a sequence of gene names.
        
        Parameters:
            encoded_tuple (tuple): A tuple of hash keys representing an encoded gene sequence.
            verbose (bool): If True, prints detailed debug information.
        
        Returns:
            list of str: A list of gene names decoded from the hash keys.
        """
        # Convert the encoded tuple back to a list for processing
        stack = list(encoded_tuple)
        decoded_sequence = []

        while stack:
            hash_key = stack.pop(0)  # Pop the first item (hash key) for decoding

            if hash_key in self.encodings:
                value = self.encodings[hash_key]

                if isinstance(value, tuple):  # Handling captured segments
                    if verbose:
                        print(f"Decompressing captured segment with hash key {hash_key}")
                    # Push the contents of the captured segment to the start of the stack for decoding
                    stack = list(value) + stack
                else:
                    # Direct mapping of hash key to gene, append the value to the decoded list
                    decoded_sequence.append(value)
                    if verbose:
                        print(f"Decoding hash key {hash_key} to '{value}'.")

            else:
                decoded_sequence.append("Unknown")
                if verbose:
                    print(f"Hash key {hash_key} is unknown.")

        return decoded_sequence



    def capture_segment(self, encoded_segment, verbose=False):
        """
        Captures a segment of encoded genes, assigning it a unique hash key.
        
        Parameters:
            encoded_segment (list of int): The segment of encoded genes to capture.
            verbose (bool): If True, prints detailed debug information.
        
        Returns:
            int: The hash key associated with the captured segment.
        """
        # Encapsulate the encoded segment in a tuple to use as a key
        captured_key = tuple(encoded_segment)
    
        # Check if this segment has already been captured by looking it up with its content
        if captured_key in self.captured_segments:
            hash_key = self.captured_segments[captured_key]
            if verbose:
                print(f"Segment {captured_key} is already captured with hash key {hash_key}.")
            return hash_key
    
        # Use the current gene_counter to assign a unique identifier to the captured segment
        unique_identifier = self.gene_counter
        # Increment the gene_counter for future use
        self.gene_counter += 1
    
        # Generate a hash key for the unique identifier of the captured segment
        hash_key = self.generate_hash_key(unique_identifier)
    
        # Store the captured segment with its content as the key and the hash key as the value
        self.captured_segments[captured_key] = hash_key
        # In the encodings, map the hash key to the captured segment's content for decoding purposes
        self.encodings[hash_key] = captured_key
    
        if verbose:
            print(f"Capturing segment {captured_key} with hash key {hash_key}.")
            print(f"Current encodings: {self.encodings}")
            print(f"Current captured segments: {self.captured_segments}")
    
        return hash_key

    
    def open_segment(self, hash_key, no_delimit=False, verbose=False):
        """
        Decompresses a captured segment, optionally including start and end delimiters.
        
        Parameters:
            hash_key (int): The hash key of the captured segment to open.
            no_delimit (bool): If True, does not include delimiters in the opened segment.
            verbose (bool): If True, prints detailed debug information.
        
        Returns:
            list of int: The decompressed list of hash keys representing the opened segment.
        """
        decompressed_codons = []
    
        # Use .get() to safely access the dictionary and avoid KeyError
        encoded_item = self.encodings.get(hash_key)
    
        # Check if the encoded_item exists and is a tuple (indicating a captured segment)
        if encoded_item and isinstance(encoded_item, tuple):
            if verbose:
                print(f"Decompressing captured segment for hash key {hash_key}.")
    
            if not no_delimit:
                # Add start delimiter if no_delimit is False
                start_delimiter_hash_key = self.reverse_encodings['Start']
                decompressed_codons.append(start_delimiter_hash_key)
    
            # Iterate through the tuple and add each hash key to the decompressed_codons list
            for gene_hash_key in encoded_item:
                decompressed_codons.append(gene_hash_key)  # gene_hash_key is already an integer hash key
    
            if not no_delimit:
                # Add end delimiter if no_delimit is False
                end_delimiter_hash_key = self.reverse_encodings['End']
                decompressed_codons.append(end_delimiter_hash_key)
        else:
            if verbose:
                print(f"Hash key {hash_key} is not a captured segment or is unknown, returning as is.")
            decompressed_codons.append(hash_key)
    
        return decompressed_codons



    
    
    def generate_random_organism(self, functional_length=100, include_specials=False, special_spacing=10, probability=0.99, verbose=False):
        """
        Generates a random organism by randomly selecting genes and potentially including special gene pairs.
        
        This method randomly selects genes from the available gene pool to construct an organism's gene sequence.
        Special gene pairs ('Start' and 'End') can be optionally included at specified intervals within the organism.

        Parameters:
            functional_length (int): The number of genes to include in the organism, excluding special genes.
            include_specials (bool): If True, includes special gene pairs ('Start' and 'End') within the organism.
            special_spacing (int): The minimum number of genes between successive special gene pairs.
            probability (float): The probability of inserting a special gene pair at any valid position.
            verbose (bool): If True, prints detailed debug information about the generation process.

        Returns:
            list of int: A list of hash keys representing the encoded organism, which includes both regular genes
                         and, optionally, special genes.
        
        Notes:
            The special genes are inserted if the random number generated is less than the specified probability and
            the spacing between them is maintained as per 'special_spacing'. The functional length of the organism
            might be extended if special genes are inserted.
        """
        gene_pool = [gene for gene in self.reverse_encodings if gene not in ['Start', 'End']]
        organism_genes = [random.choice(gene_pool) for _ in range(functional_length)]
        special_gene_indices = set()
    
        if include_specials:
            for i in range(len(organism_genes)):
                if random.random() <= probability:
                    if all(abs(i - idx) >= special_spacing for idx in special_gene_indices):
                        organism_genes.insert(i, 'Start')
                        end_index = min(i + special_spacing, len(organism_genes))
                        organism_genes.insert(end_index, 'End')
                        special_gene_indices.update([i, end_index])
                        print(organism_genes)
    
        encoded_organism = self.encode(organism_genes, verbose = verbose)  # Pass list directly
    
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
        
        # Encode a list of genes, including 'End' as part of the list
        for gene in genes:
            encoded = tuple(self.manager.encode([gene, 'End'], verbose=True))
            decoded = self.manager.decode(encoded, verbose=True)
            # Expected decoded string should include 'End' as a separate element
            expected_decoded_str = gene + ' End'
            decoded_str = ' '.join(decoded)
            self.assertEqual(decoded_str, expected_decoded_str, f"Encoded sequence should decode back to '{expected_decoded_str}'.")

    def test_decode_unknown_hash_key(self):
        # Generate a hash key that is likely not in use by adding a unique gene and then incrementing the counter
        self.manager.add_gene('UniqueGene', verbose=False)
        unknown_hash_key = self.manager.gene_counter + 1  # Assuming gene_counter is still accessible in this context

        # Ensure the argument is a tuple
        encoded = tuple([unknown_hash_key])

        # Call decode with the correct tuple argument
        decoded = self.manager.decode(encoded, verbose=True)

        # Check if 'Unknown' is in the decoded list
        self.assertIn('Unknown', decoded, "Unknown hash key should decode to 'Unknown'.")



class TestEncodingManagerCapturedSegments(unittest.TestCase):
    def setUp(self):
        self.manager = EncodingManager()

    def test_capture_and_decode(self):
        self.manager.add_gene('A')
        self.manager.add_gene('B')

        # The encode method should return a list of hash keys; ensure it's being called correctly
        encoded_segment = self.manager.encode(['A', 'B'],
                                              verbose=True)  # Ensure 'A' and 'B' are in a list if they represent individual genes

        captured_codon = self.manager.capture_segment(encoded_segment, verbose=True)

        # Wrap captured_codon in a list, then convert to tuple for decode
        decoded_sequence = self.manager.decode(tuple([captured_codon]), verbose=True)

        # Join the decoded list to form a string for comparison
        decoded_str = ' '.join(decoded_sequence)
        self.assertEqual(decoded_str, 'A B', "The decoded sequence should match the original segment.")

    def test_explicit_nested_capture_and_decoding(self):
        genes = ['1', '2', '3', '4', '5']
        for gene in genes:
            self.manager.add_gene(gene, verbose=True)

        encoded_segment = []
        for gene in genes:
            # Ensure each gene is passed as a single-element list to encode
            encoded_gene = self.manager.encode([gene], verbose=True)
            encoded_segment.extend(encoded_gene)

            # Capture the current segment
            captured_codon = self.manager.capture_segment(encoded_segment, verbose=True)

            # Decode using a tuple containing the captured codon
            decoded_sequence = self.manager.decode(tuple([captured_codon]), verbose=True)

            # Join the decoded list to form a string for comparison
            decoded_str = ' '.join(decoded_sequence)

            # The expected decoded string should match the sequence of genes captured so far
            expected_decoded_str = ' '.join(genes[:len(decoded_sequence)])
            self.assertEqual(decoded_str, expected_decoded_str, f"Decoded sequence should match {expected_decoded_str}")

    def test_duplicate_segment_capture(self):
        self.manager.add_gene('X')
        self.manager.add_gene('Y')

        # Ensure genes are passed as a list to the encode method
        encoded_segment_1 = self.manager.encode(['X', 'Y'], verbose=True)
        captured_codon_1 = self.manager.capture_segment(encoded_segment_1, verbose=True)

        encoded_segment_2 = self.manager.encode(['X', 'Y'], verbose=True)
        captured_codon_2 = self.manager.capture_segment(encoded_segment_2, verbose=True)

        # Verify that the same hash key is reused for the duplicate segment
        self.assertEqual(captured_codon_1, captured_codon_2, "Duplicate segments should reuse the same hash key.")

        # Convert captured codon to a tuple and decode
        decoded_sequence_1 = self.manager.decode(tuple([captured_codon_1]), verbose=True)
        decoded_sequence_2 = self.manager.decode(tuple([captured_codon_2]), verbose=True)

        # Join the decoded list to form a string for comparison
        decoded_str_1 = ' '.join(decoded_sequence_1)
        decoded_str_2 = ' '.join(decoded_sequence_2)

        self.assertEqual(decoded_str_1, 'X Y', "The decoded sequence should match the original segment 'X Y'.")
        self.assertEqual(decoded_str_1, decoded_str_2, "Decoded sequences from duplicate captures should be identical.")



class TestEncodingManagerNestedCaptures(unittest.TestCase):
    def setUp(self):
        self.manager = EncodingManager()

    def test_nested_segment_capture_with_multiple_genes(self):
        genes = ['X', 'Y', 'Z', 'W']
        for gene in genes:
            self.manager.add_gene(gene, verbose=True)

        # Encode and capture the initial segment 'X Y'. Ensure genes are passed as a list to encode.
        initial_encoded_segment = self.manager.encode(['X', 'Y'], verbose=True)
        initial_capture_codon = self.manager.capture_segment(initial_encoded_segment, verbose=True)

        # Encode the next part 'Z W' and create a nested segment that includes the hash key of the initial captured segment
        next_encoded_segment = self.manager.encode(['Z', 'W'], verbose=True)
        nested_encoded_segment = [initial_capture_codon] + next_encoded_segment

        # Capture the nested segment
        nested_capture_codon = self.manager.capture_segment(nested_encoded_segment, verbose=True)

        # Decode the nested capture to test if the nested structure is preserved. Ensure captured codon is in a tuple.
        decoded_nested_sequence = self.manager.decode(tuple([nested_capture_codon]), verbose=True)

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
        self.single_captured_hash_key = self.encoding_manager.capture_segment(self.single_segment)
        # For nested capture, include the captured hash key in a new segment with 'D'
        self.nested_segment = [self.single_captured_hash_key] + [self.encoding_manager.encode(self.genes[3])[0]]
        self.nested_captured_hash_key = self.encoding_manager.capture_segment(self.nested_segment)

    def test_single_level_capture_and_open_with_delimiters(self):
        # Open with delimiters
        opened_segment = self.encoding_manager.open_segment(self.single_captured_hash_key, no_delimit=False)
        expected_segment = [self.encoding_manager.reverse_encodings['Start']] + self.single_segment + [self.encoding_manager.reverse_encodings['End']]
        self.assertEqual(opened_segment, expected_segment, "Opened segment with delimiters does not match expected.")

    def test_single_level_capture_and_open_without_delimiters(self):
        # Open without delimiters
        opened_segment = self.encoding_manager.open_segment(self.single_captured_hash_key, no_delimit=True)
        self.assertEqual(opened_segment, self.single_segment, "Opened segment without delimiters does not match expected.")

    def test_nested_capture_and_open(self):
        # This test checks that open_segment only opens the next layer of nesting
        genes = ['X', 'Y', 'Z', 'W']
        for gene in genes:
            self.encoding_manager.add_gene(gene, verbose=True)
        
        # Encode and capture the initial segment 'X Y'
        initial_encoded_segment = self.encoding_manager.encode('X Y', verbose=True)
        initial_capture_hash_key = self.encoding_manager.capture_segment(initial_encoded_segment, verbose=True)
        
        # Create a nested segment that includes the hash key of the initial captured segment
        nested_encoded_segment = [initial_capture_hash_key] + self.encoding_manager.encode('Z W', verbose=True)
        
        # Capture the nested segment
        nested_capture_hash_key = self.encoding_manager.capture_segment(nested_encoded_segment, verbose=True)
        
        # Open the nested capture to check if only the next layer is decompressed
        opened_nested_segment = self.encoding_manager.open_segment(nested_capture_hash_key, no_delimit=True)
        
        # Expected behavior is to decompress only the next layer, showing the initial capture hash key followed by 'Z' and 'W'
        expected_nested_segment = [initial_capture_hash_key] + [self.encoding_manager.encode(gene)[0] for gene in genes[2:]]
        
        self.assertEqual(opened_nested_segment, expected_nested_segment, "Opened nested segment should decompress only the next layer.")




class TestEncodingIntegrationAndGeneSelection(unittest.TestCase):
    def setUp(self):
        # Initialize the EncodingManager with predefined encodings using hash keys
        self.initial_manager = EncodingManager()
        self.genes = ['A', 'B', 'C', 'D']
        for gene in self.genes:
            self.initial_manager.add_gene(gene)
        # Encode genes to create segments and capture one to test integration
        self.segment = [self.initial_manager.encode(gene)[0] for gene in self.genes[:2]]
        self.captured_hash_key = self.initial_manager.capture_segment(self.segment)
        # Define the probability of selecting a base gene over a captured segment
        self.base_gene_prob = 0.8  # You can adjust this value as needed

    def select_gene(self, manager):
        # Select a gene based on a probability, favoring base genes but occasionally choosing captured segments
        if random.random() < self.base_gene_prob or not manager.captured_segments:
            base_gene = random.choice(self.genes)
            gene_hash_key = manager.reverse_encodings[base_gene]
        else:
            captured_segment = random.choice(list(manager.captured_segments.keys()))
            gene_hash_key = manager.captured_segments[captured_segment]
        return gene_hash_key

    def test_integration_and_gene_selection(self):
        # Reinitialize EncodingManager and integrate encodings, including base genes and captured segments
        new_manager = EncodingManager()
        new_manager.integrate_uploaded_encodings(self.initial_manager.encodings, self.genes)

        # Test gene selection to cover both base genes and captured segments
        for _ in range(10):
            selected_gene_hash_key = self.select_gene(new_manager)
            self.assertTrue(
                selected_gene_hash_key in new_manager.encodings or
                selected_gene_hash_key in new_manager.captured_segments.values(),
                "Selected gene should be from either base genes or captured segments."
            )



if __name__ == '__main__':
    unittest.main(verbosity=True)
