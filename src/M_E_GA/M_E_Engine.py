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
        self.captured_segments = {}
        self.gene_counter = 3  # Start the counter from 3 after 'Start' and 'End'

        # Add default delimiters with predefined unique IDs
        self.add_gene('Start', predefined_id=1)
        self.add_gene('End', predefined_id=2)

    def generate_hash_key(self, identifier):
        # Use xxhash's 64-bit version to generate a longer hash
        return xxhash.xxh64_intdigest(str(identifier))

    def add_gene(self, gene, verbose=False, predefined_id=None):
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
                    print(
                        f"Skipping gene '{value}' with key '{key}' as it does not match expected base genes or default genes.")

        # Update gene counter to avoid conflicts
        max_hash_key = max(self.encodings.keys(), default=0)
        self.gene_counter = max(self.gene_counter, max_hash_key + 1)
        if verbose:
            print("Final updated gene counter:", self.gene_counter)

    def encode(self, genes, verbose=False):
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

    def generate_random_organism(self, functional_length=100, include_specials=False, special_spacing=10,
                                 probability=0.99, verbose=False):
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

        encoded_organism = self.encode(organism_genes, verbose=verbose)  # Pass list directly

        if verbose:
            print("Generated Encoded Organism:", encoded_organism)

        return encoded_organism


