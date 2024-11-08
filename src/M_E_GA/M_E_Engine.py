# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:48:15 2024

@author: Matt Andrews
"""
# GNU GENERAL PUBLIC LICENSE
# By running this code, you acknowledge and agree to the terms of the LICENSE file
# provided in the repository. 


import random
import xxhash
import functools


class EncodingManager:
    def __init__(self):
        # Initialize with default genes 'Start' and 'End'
        self.encodings = {}
        self.reverse_encodings = {}
        self.meta_genes = []  # Renamed from captured_segments
        self.gene_counter = 3  # Start the counter from 3 after 'Start' and 'End'

        # Add default delimiters with predefined unique IDs
        self.add_gene('Start', predefined_id=1)
        self.add_gene('End', predefined_id=2)

    def generate_hash_key(self, identifier):
        # Use xxhash's 64-bit version to generate a longer hash
        return xxhash.xxh64_intdigest(str(identifier))

    def add_gene(self, gene, verbose=False, predefined_id=None):
        """
        Adds a new gene to the encodings.

        Args:
            gene (str): The gene to add.
            verbose (bool): If True, prints confirmation.
            predefined_id (int, optional): If provided, uses this as the hash key identifier.
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
        Integrates uploaded encodings into the existing encoding manager.

        Args:
            uploaded_encodings (dict or str): Encodings to integrate.
            base_genes (list): List of base genes.
            verbose (bool): If True, prints detailed information.
        """
        if verbose:
            print("Starting integration of uploaded encodings...")

        if isinstance(uploaded_encodings, str):
            # Parse the string into a dictionary assuming format "key:value,key:value"
            uploaded_encodings = {int(k): v for k, v in (item.split(':') for item in uploaded_encodings.split(','))}
            if verbose:
                print("Uploaded encodings after parsing:", uploaded_encodings)

        # Identify the hash keys for default genes 'Start' and 'End' from the initial manager
        start_key = self.reverse_encodings.get('Start')
        end_key = self.reverse_encodings.get('End')
        if verbose:
            print(f"Default gene 'Start' hash key: {start_key}, 'End' hash key: {end_key}")

        # Integrate base and default genes along with meta genes
        for key, value in uploaded_encodings.items():
            if value in base_genes or key in [start_key, end_key]:
                if value not in self.reverse_encodings or key in [start_key, end_key]:
                    self.encodings[key] = value
                    self.reverse_encodings[value] = key
                    if verbose:
                        print(f"Integrated gene '{value}' with key '{key}'.")
            elif isinstance(value, tuple):  # Handle meta genes
                self.encodings[key] = value
                self.meta_genes.append(key)  # Append to the list
                if verbose:
                    print(f"Integrated meta gene '{value}' with key '{key}'.")
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
        Encodes a list of genes into their corresponding hash keys.

        Args:
            genes (list): List of gene strings to encode.
            verbose (bool): If True, prints encoding details.

        Returns:
            list: List of hash keys representing the encoded genes.
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
        Decodes a tuple of hash keys back into their gene sequences.

        Args:
            encoded_tuple (tuple): Tuple of hash keys to decode.
            verbose (bool): If True, prints decoding details.

        Returns:
            list: List of decoded gene strings.
        """
        # Convert the encoded tuple back to a list for processing
        stack = list(encoded_tuple)
        decoded_sequence = []

        while stack:
            hash_key = stack.pop(0)  # Pop the first item (hash key) for decoding

            if hash_key in self.encodings:
                value = self.encodings[hash_key]

                if isinstance(value, tuple):  # Handling meta genes
                    if verbose:
                        print(f"Decompressing meta gene with hash key {hash_key}")
                    # Push the contents of the meta gene to the start of the stack for decoding
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

    def capture_metagene(self, encoded_segment, verbose=False):
        """
        Captures a segment of encoded genes as a Meta Gene.

        Args:
            encoded_segment (list): List of hash keys representing the segment to capture.
            verbose (bool): If True, prints capture details.

        Returns:
            int: Hash key assigned to the captured Meta Gene.
        """
        # Always assign a new unique hash key for each capture
        unique_identifier = self.gene_counter
        hash_key = self.generate_hash_key(unique_identifier)
        self.gene_counter += 1

        # Map the hash_key to the encoded_segment
        self.encodings[hash_key] = tuple(encoded_segment)
        self.meta_genes.append(hash_key)  # Append to the list

        if verbose:
            print(f"Captured Meta Gene {encoded_segment} with hash key {hash_key}.")

        return hash_key

    def open_metagene(self, hash_key, no_delimit=False, verbose=False):
        """
        Opens a captured Meta Gene, decompressing it back into its gene sequence.

        Args:
            hash_key (int): Hash key of the Meta Gene to open.
            no_delimit (bool): If True, omits adding 'Start' and 'End' delimiters.
            verbose (bool): If True, prints decompression details.

        Returns:
            list: List of hash keys representing the decompressed gene sequence.
        """
        decompressed_codons = []

        # Use .get() to safely access the dictionary and avoid KeyError
        encoded_item = self.encodings.get(hash_key)

        # Check if the encoded_item exists and is a tuple (indicating a meta gene)
        if encoded_item and isinstance(encoded_item, tuple):
            if verbose:
                print(f"Decompressing meta gene for hash key {hash_key}.")

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
                print(f"Hash key {hash_key} is not a meta gene or is unknown, returning as is.")
            decompressed_codons.append(hash_key)

        return decompressed_codons

    def generate_random_organism(self, functional_length=100, include_specials=False, special_spacing=10,
                                 probability=0.99, verbose=False):
        """
        Generates a random organism with optional special delimiters.

        Args:
            functional_length (int): Number of functional genes.
            include_specials (bool): If True, includes 'Start' and 'End' delimiters.
            special_spacing (int): Minimum spacing between special delimiters.
            probability (float): Probability of inserting a delimiter.
            verbose (bool): If True, prints generation details.

        Returns:
            list: List of hash keys representing the encoded organism.
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
                        if verbose:
                            print(organism_genes)

        encoded_organism = self.encode(organism_genes, verbose=verbose)  # Pass list directly

        if verbose:
            print("Generated Encoded Organism:", encoded_organism)

        return encoded_organism
