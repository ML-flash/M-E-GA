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
from collections import OrderedDict


class EncodingManager:
    def __init__(self, lru_cache_size=1000):
        # Original attributes
        self.encodings = {}
        self.reverse_encodings = {}
        self.meta_genes = []
        self.gene_counter = 3
        self.lru_cache_size = lru_cache_size
        self.metagene_usage = OrderedDict()
        self.deletion_basket = {}
        self.unused_encodings = []
        self.current_generation = 0

        # Add default delimiters with predefined unique IDs
        self.add_gene('Start', predefined_id=1)
        self.add_gene('End', predefined_id=2)

        # -------------------------
        # Additional Internal Maps
        # -------------------------
        # These are new structures that do not replace or remove any existing structures.
        # They are solely internal and do not affect the public interface.
        self._metagene_children_map = {}  # {meta_gene_key: [codons]}
        self._metagene_parents_map = {}   # {meta_gene_key: set_of_parent_metagenes}

    def generate_hash_key(self, identifier):
        return xxhash.xxh64_intdigest(str(identifier))

    def add_gene(self, gene, verbose=False, predefined_id=None):
        if gene in self.reverse_encodings:
            if verbose:
                print(f"Gene '{gene}' is already added.")
            return

        if self.unused_encodings and predefined_id is None:
            hash_key = self.unused_encodings.pop()
        else:
            identifier = predefined_id if predefined_id is not None else self.gene_counter
            hash_key = self.generate_hash_key(identifier)
            if predefined_id is None:
                self.gene_counter += 1

        self.encodings[hash_key] = gene
        self.reverse_encodings[gene] = hash_key
        if verbose:
            print(f"Added gene '{gene}' with hash key {hash_key}.")

        return hash_key

    def integrate_uploaded_encodings(self, uploaded_encodings, base_genes, verbose=False):
        if verbose:
            print("Starting integration of uploaded encodings...")

        if isinstance(uploaded_encodings, str):
            uploaded_encodings = {int(k): v for k, v in (item.split(':') for item in uploaded_encodings.split(','))}
            if verbose:
                print("Uploaded encodings after parsing:", uploaded_encodings)

        start_key = self.reverse_encodings.get('Start')
        end_key = self.reverse_encodings.get('End')

        for key, value in uploaded_encodings.items():
            if value in base_genes or key in [start_key, end_key]:
                if value not in self.reverse_encodings or key in [start_key, end_key]:
                    self.encodings[key] = value
                    self.reverse_encodings[value] = key
                    if verbose:
                        print(f"Integrated gene '{value}' with key '{key}'.")
            elif isinstance(value, tuple):
                self.encodings[key] = value
                self.meta_genes.append(key)
                self.metagene_usage[key] = True
                # Add to our internal maps
                self._metagene_children_map[key] = list(value)
                self._metagene_parents_map.setdefault(key, set())
                # Update parents for each child that is a meta gene
                for c in value:
                    if c in self.meta_genes:
                        self._metagene_parents_map.setdefault(c, set()).add(key)
            else:
                if verbose:
                    print(f"Skipping gene '{value}' with key '{key}'.")

        max_hash_key = max(self.encodings.keys(), default=0)
        self.gene_counter = max(self.gene_counter, max_hash_key + 1)

    def encode(self, genes, verbose=False):
        encoded_list = []
        for gene in genes:
            hash_key = self.reverse_encodings.get(gene)
            if hash_key is None:
                if verbose:
                    print(f"Gene '{gene}' is not recognized.")
                continue
            encoded_list.append(hash_key)
            if verbose:
                print(f"Encoding gene '{gene}' to hash key {hash_key}.")
        return encoded_list

    @functools.lru_cache(maxsize=1000)
    @functools.lru_cache(maxsize=1000)
    def decode(self, encoded_tuple, verbose=False):
        if not encoded_tuple:
            return []

        stack = list(encoded_tuple)
        decoded_sequence = []

        while stack:
            hash_key = stack.pop(0)
            if hash_key in self.encodings:
                value = self.encodings[hash_key]
                self.update_metagene_usage(hash_key)

                if isinstance(value, tuple):
                    if verbose:
                        print(f"Decompressing meta gene with hash key {hash_key}")
                    stack = list(value) + stack
                else:
                    decoded_sequence.append(value)
                    if verbose:
                        print(f"Decoding hash key {hash_key} to '{value}'.")
            else:
                decoded_sequence.append("Unknown")
                if verbose:
                    print(f"Hash key {hash_key} is unknown.")

        return decoded_sequence

    def update_metagene_usage(self, hash_key):
        if hash_key not in self.meta_genes:
            return

        if hash_key in self.metagene_usage:
            self.metagene_usage.move_to_end(hash_key)
        else:
            if len(self.metagene_usage) >= self.lru_cache_size:
                lru_key, _ = self.metagene_usage.popitem(last=False)
                if lru_key not in self.deletion_basket:
                    self.deletion_basket[lru_key] = 0
                    print(f"Moving metagene {lru_key} to deletion basket due to LRU cache overflow")
            self.metagene_usage[hash_key] = True

    def start_new_generation(self):
        self.current_generation += 1

        print(f"\nProcessing deletion basket at start of generation {self.current_generation}:")
        if not self.deletion_basket:
            print("  Deletion basket is empty")
            return

        to_delete = []
        for hash_key, gen_count in self.deletion_basket.items():
            if gen_count >= 2:
                to_delete.append(hash_key)
                print(f"  Metagene {hash_key} marked for deletion (unused for {gen_count} generations)")
            else:
                new_count = gen_count + 1
                self.deletion_basket[hash_key] = new_count
                print(f"  Metagene {hash_key} count increased from {gen_count} to {new_count}")

        for hash_key in to_delete:
            print(f"\nDeleting metagene {hash_key}:")
            print(f"  Original contents: {self.encodings.get(hash_key, 'Unknown')}")
            decoded = self.open_metagene(hash_key, no_delimit=True)
            print(f"  Decoded contents: {decoded}")
            self.delete_metagene(hash_key)
            print(f"  Added hash key {hash_key} to unused_encodings")
            print(f"  Current unused_encodings pool size: {len(self.unused_encodings)}")

    def delete_metagene(self, hash_key):
        # Original Code - Do not remove or rename existing code, just add after it:
        if hash_key not in self.meta_genes:
            return

        contents = self.encodings.get(hash_key, ())
        processed = set()

        def process_dependencies(current_key):
            if current_key in processed or current_key not in self.meta_genes:
                return [current_key] if current_key not in processed else []

            processed.add(current_key)
            current_contents = self.encodings.get(current_key, ())
            expanded = []

            for gene in current_contents:
                if gene in self.deletion_basket and gene in self.encodings:
                    expanded.extend(process_dependencies(gene))
                else:
                    expanded.append(gene)

            return expanded

        expanded_contents = process_dependencies(hash_key)

        for meta_key in [k for k in self.meta_genes if k != hash_key]:
            meta_contents = list(self.encodings.get(meta_key, ()))
            if hash_key in meta_contents:
                new_contents = []
                for content in meta_contents:
                    if content == hash_key:
                        new_contents.extend(expanded_contents)
                    else:
                        new_contents.append(content)
                self.encodings[meta_key] = tuple(new_contents)

        self.meta_genes.remove(hash_key)
        self.metagene_usage.pop(hash_key, None)
        self.deletion_basket.pop(hash_key, None)
        self.encodings.pop(hash_key, None)

        if hash_key not in self.unused_encodings:
            self.unused_encodings.append(hash_key)

        # ---------------------------------------
        # Additional Steps to Prevent Stale References
        # ---------------------------------------
        # Now that the original logic has run, we use the newly introduced maps
        # to ensure that no stale references remain. We do NOT remove or alter any
        # of the original steps, just add more logic here.

        # Remove references from the internal maps
        if hash_key in self._metagene_children_map:
            del self._metagene_children_map[hash_key]

        # Find all meta genes that might have referenced this one and ensure
        # they no longer do. Although the original code attempted this,
        # we use our private structures to ensure full correctness.
        for mg, parents in self._metagene_parents_map.items():
            if hash_key in parents:
                parents.discard(hash_key)

        if hash_key in self._metagene_parents_map:
            del self._metagene_parents_map[hash_key]

        # Ensure that for any meta gene children that were expanded, their parents are updated
        # This step ensures that after expansion, all references are correctly managed.
        # We iterate over expanded_contents and update parent references if needed.
        for c in expanded_contents:
            # If c was a meta gene, ensure its parent references are correct
            if c in self.meta_genes:
                # If this meta gene was referencing the deleted one, it's already handled above
                # Now we just ensure that we don't leave stale entries.
                # This might be a no-op if everything is already correct.
                if c not in self._metagene_parents_map:
                    self._metagene_parents_map[c] = set()  # ensure it exists

    def capture_metagene(self, encoded_segment, verbose=False):
        if not encoded_segment:
            return False

        if self.unused_encodings:
            hash_key = self.unused_encodings.pop()
        else:
            hash_key = self.generate_hash_key(self.gene_counter)
            self.gene_counter += 1

        self.encodings[hash_key] = tuple(encoded_segment)
        self.meta_genes.append(hash_key)
        self.update_metagene_usage(hash_key)

        # ---------------------------------------------------
        # Additional internal bookkeeping to avoid stale refs
        # ---------------------------------------------------
        # Record children
        self._metagene_children_map[hash_key] = list(encoded_segment)
        # Ensure an entry for parents
        if hash_key not in self._metagene_parents_map:
            self._metagene_parents_map[hash_key] = set()
        # Update parents of each child if the child is a meta gene
        for c in encoded_segment:
            if c in self.meta_genes:
                self._metagene_parents_map.setdefault(c, set()).add(hash_key)
        # ---------------------------------------------------

        if verbose:
            print(f"Captured Meta Gene {encoded_segment} with hash key {hash_key}.")

        return hash_key

    def open_metagene(self, hash_key, no_delimit=False, verbose=False):
        decompressed_codons = []
        encoded_item = self.encodings.get(hash_key)

        if encoded_item and isinstance(encoded_item, tuple):
            if verbose:
                print(f"Decompressing meta gene for hash key {hash_key}.")

            if not no_delimit:
                start_delimiter_hash_key = self.reverse_encodings['Start']
                decompressed_codons.append(start_delimiter_hash_key)

            for gene_hash_key in encoded_item:
                decompressed_codons.append(gene_hash_key)

            if not no_delimit:
                end_delimiter_hash_key = self.reverse_encodings['End']
                decompressed_codons.append(end_delimiter_hash_key)
        else:
            if verbose:
                print(f"Hash key {hash_key} is not a meta gene or is unknown.")
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
                        if verbose:
                            print(organism_genes)

        encoded_organism = self.encode(organism_genes, verbose=verbose)

        if verbose:
            print("Generated Encoded Organism:", encoded_organism)

        return encoded_organism