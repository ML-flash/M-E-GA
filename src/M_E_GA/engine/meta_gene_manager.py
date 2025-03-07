"""
meta_gene_manager.py

Handles metagene operations: capturing segments, opening them,
deleting them, and managing LRU usage + deletion_basket.

Part of the refactoring for Single Responsibility Principle.
"""

import random
from collections import OrderedDict


class MetaGeneManager:
    """
    Manages capturing, opening, and deleting metagenes,
    along with LRU usage tracking and the deletion basket.

    Attributes:
        meta_genes (list): A list of hash keys that represent metagenes.
        meta_gene_stack (list): Ordered from oldest to newest metagenes.
        metagene_usage (OrderedDict): LRU data structure for usage tracking.
        deletion_basket (dict): Maps metagene hash keys to how many generations they've been unused.
        lru_cache_size (int): Max number of active items in the LRU cache before eviction.
        current_generation (int): Tracks which generation we're in to manage deletions.
        debug (bool): Flag for debug prints.
    """

    def __init__(
            self,
            encodings,
            meta_genes,
            meta_gene_stack,
            metagene_usage,
            deletion_basket,
            lru_cache_size=100,
            debug=False
    ):
        """
        Initialize the MetaGeneManager.

        :param encodings: The global encodings dict (hash_key -> content).
        :param meta_genes: The global list of meta-gene hash keys.
        :param meta_gene_stack: The stack (list) to track metagenes in order of creation.
        :param metagene_usage: The LRU usage dictionary.
        :param deletion_basket: The dictionary that tracks how long a metagene is unused.
        :param lru_cache_size: The maximum size of the LRU cache.
        :param debug: Whether to enable debug logs.
        """
        self.encodings = encodings
        self.meta_genes = meta_genes
        self.meta_gene_stack = meta_gene_stack
        self.metagene_usage = metagene_usage
        self.deletion_basket = deletion_basket
        self.lru_cache_size = lru_cache_size
        self.current_generation = 0
        self.debug = debug
        # Reference to unused_encodings will be set by EncodingManager
        self.unused_encodings = None

    def get_metagene_status(self):
        """
        Retrieve a status report for current metagene usage.

        :return: A dict with generation, total metagenes, how many in basket, unused, etc.
        """
        # We'll sample from meta_genes if debug
        samples = []
        if self.debug and self.meta_genes:
            sample_size = min(3, len(self.meta_genes))
            if sample_size > 0:
                for hash_key in random.sample(self.meta_genes, sample_size):
                    if hash_key in self.encodings:
                        samples.append({
                            'id': hash_key,
                            'content': self.encodings[hash_key],
                            'in_lru': hash_key in self.metagene_usage,
                            'in_basket': hash_key in self.deletion_basket,
                        })

        return {
            'generation': self.current_generation,
            'total_metagenes': len(self.meta_genes),
            'in_basket': len(self.deletion_basket),
            'in_lru': len(self.metagene_usage),
            'samples': samples
        }

    def start_new_generation(self):
        """
        Increase generation count, increment usage counters in deletion basket,
        and remove any that exceed threshold.
        """
        self.current_generation += 1
        to_delete = []

        # Identify which metagenes to delete
        for hash_key, gen_count in list(self.deletion_basket.items()):
            if gen_count >= 2:
                to_delete.append(hash_key)
                if self.debug:
                    print(
                        f"[MetaGeneManager] Marking metagene {hash_key} for deletion. unused for {gen_count} generations.")
            else:
                self.deletion_basket[hash_key] = gen_count + 1
                if self.debug:
                    print(f"[MetaGeneManager] Incrementing usage for metagene {hash_key} to {gen_count + 1}.")

        # Do actual deletions without special ordering - relies on proper replacement in delete_metagene
        for hash_key in to_delete:
            self.delete_metagene(hash_key)

    def update_metagene_usage(self, hash_key):
        """
        Track usage of a particular metagene in the LRU structure.

        :param hash_key: The hash key of the metagene in question.
        """
        if hash_key not in self.meta_genes:
            return

        # If it's in the basket, remove it since it's no longer unused
        self.deletion_basket.pop(hash_key, None)

        # Bump it to the end of the LRU
        if hash_key in self.metagene_usage:
            self.metagene_usage.move_to_end(hash_key)
        else:
            if len(self.metagene_usage) >= self.lru_cache_size:
                # Evict the least recently used
                lru_key, _ = self.metagene_usage.popitem(last=False)
                if lru_key not in self.deletion_basket:
                    self.deletion_basket[lru_key] = 0
            self.metagene_usage[hash_key] = True

    def add_meta_gene(self, hash_key):
        """
        Add a newly minted metagene to the main lists.

        :param hash_key: The integer key for the new metagene.
        """
        if hash_key not in self.meta_genes:
            self.meta_genes.append(hash_key)
        if hash_key not in self.meta_gene_stack:
            self.meta_gene_stack.append(hash_key)

    def remove_meta_gene(self, hash_key):
        """
        Remove a metagene from the main lists.

        :param hash_key: The integer key for the metagene to remove.
        """
        if hash_key in self.meta_genes:
            self.meta_genes.remove(hash_key)
        if hash_key in self.meta_gene_stack:
            self.meta_gene_stack.remove(hash_key)

    def delete_metagene(self, hash_key):
        """
        Delete a metagene from encodings. Inlines references within other metagenes
        that point to the soon-to-be-deleted one.

        :param hash_key: The integer key for the metagene to delete.
        """
        if hash_key not in self.meta_genes or hash_key not in self.encodings:
            return

        # Get the content of the metagene that will be deleted
        target_contents = list(self.encodings[hash_key])

        # Replace references in all other metagenes
        for meta_key in list(self.meta_genes):  # Create a copy of the list to safely iterate
            if meta_key == hash_key:
                continue  # Skip the metagene being deleted
                
            if meta_key not in self.encodings or not isinstance(self.encodings[meta_key], tuple):
                continue  # Skip invalid metagenes
                
            meta_contents = list(self.encodings[meta_key])
            modified = False

            # Look for references to the hash_key being deleted
            i = 0
            while i < len(meta_contents):
                if meta_contents[i] == hash_key:
                    # Replace the reference with the actual contents
                    meta_contents[i:i + 1] = target_contents
                    modified = True
                    i += len(target_contents)
                else:
                    i += 1

            if modified:
                self.encodings[meta_key] = tuple(meta_contents)

        # Finally remove references
        self.remove_meta_gene(hash_key)
        self.metagene_usage.pop(hash_key, None)
        self.deletion_basket.pop(hash_key, None)
        self.encodings.pop(hash_key, None)

        # Add to unused_encodings for potential reuse
        if hasattr(self, 'unused_encodings') and self.unused_encodings is not None:
            self.unused_encodings.append(hash_key)

        if self.debug:
            print(f"[MetaGeneManager] Deleted metagene {hash_key}.")

    def capture_metagene(self, encoded_segment, generate_hash_key_func, unused_encodings, gene_counter_ref, logger=None,
                         verbose=False):
        """
        If the segment isn't empty, convert it into a new metagene or reuse
        an existing one if it's identical.

        :param encoded_segment: The actual list of hash keys representing the segment to capture.
        :param generate_hash_key_func: The function used to create new hash keys (64-bit int).
        :param unused_encodings: A list of freed hash keys to reuse, if available.
        :param gene_counter_ref: A mutable reference storing the current gene counter (list of 1 int).
        :param logger: Optional logger for logging events.
        :param verbose: If True, prints debug messages.
        :return: The integer hash key of the captured or existing metagene, or False if empty.
        """
        if not encoded_segment:
            return False

        # Store reference to unused_encodings for later use in delete_metagene
        self.unused_encodings = unused_encodings

        segment_tuple = tuple(encoded_segment)

        # Check for existing identical metagene
        for meta_id in self.meta_genes:
            if meta_id in self.encodings and self.encodings[meta_id] == segment_tuple:
                self.update_metagene_usage(meta_id)
                return meta_id

        # Need a new hash key
        if unused_encodings:
            hash_key = unused_encodings.pop(0)
            if logger:
                logger.log_event("metagene_reused", {
                    "hash_key": hash_key,
                    "generation": self.current_generation,
                    "unused_pool_size": len(unused_encodings)
                })
        else:
            new_id = gene_counter_ref[0]
            gene_counter_ref[0] += 1
            hash_key = generate_hash_key_func(new_id)

        # Store the new metagene
        self.encodings[hash_key] = segment_tuple
        self.add_meta_gene(hash_key)
        self.update_metagene_usage(hash_key)

        if verbose and self.debug:
            print(f"[MetaGeneManager] Captured new metagene with ID {hash_key}.")

        if logger:
            logger.log_event("metagene_captured", {
                "hash_key": hash_key,
                "segment": segment_tuple,
                "generation": self.current_generation
            })

        return hash_key

    def open_metagene(self, hash_key, start_key, end_key, no_delimit=False):
        """
        Decompress or 'open' a metagene by substituting its contents in place,
        optionally wrapping them with Start/End delimiters.

        :param hash_key: The integer key referencing the metagene.
        :param start_key: The integer key representing 'Start'.
        :param end_key: The integer key representing 'End'.
        :param no_delimit: If True, do not add Start/End.
        :return: A list of integer hash keys representing the expanded sequence.
        """
        if hash_key not in self.encodings or not isinstance(self.encodings[hash_key], tuple):
            return [hash_key]

        decompressed = []
        if not no_delimit:
            decompressed.append(start_key)

        decompressed.extend(self.encodings[hash_key])

        if not no_delimit:
            decompressed.append(end_key)

        return decompressed