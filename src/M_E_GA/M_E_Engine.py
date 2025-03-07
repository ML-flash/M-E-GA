"""
M_E_Engine.py

Now primarily a facade that wraps GeneManager, MetaGeneManager,
and OrganismGenerator to maintain legacy naming.

Implements the EncodingManager class that orchestrates everything.
"""

from collections import OrderedDict

import xxhash

from .engine.gene_manager import GeneManager
from .engine.meta_gene_manager import MetaGeneManager
from .engine.organism_generator import OrganismGenerator


class EncodingManager:
    """
    Orchestrates gene and metagene operations, providing a single point
    of interaction for adding genes, capturing metagenes, decoding, and
    random organism generation.

    This class was refactored to delegate single responsibilities to
    specialized managers, adhering to the SRP principle.
    """

    def __init__(self, lru_cache_size=100, logger=None, debug=False):
        """
        Initialize the EncodingManager with sub-managers.

        :param lru_cache_size: Max size for the LRU usage.
        :param logger: Optional GA_Logger or similar for logging events.
        :param debug: Whether to enable debug messages throughout sub-managers.
        """
        self.logger = logger
        self.debug = debug

        # Core shared data structures
        self.encodings = {}
        self.reverse_encodings = {}
        self.meta_genes = []
        self.meta_gene_stack = []
        self.metagene_usage = OrderedDict()
        self.deletion_basket = {}
        self.unused_encodings = []

        # We track a single integer in a list, so it can be passed by reference
        self.gene_counter_ref = [3]  # We start at 3 because 1 & 2 are for Start, End

        # Instantiate sub-managers
        self.gene_manager = GeneManager(
            encodings=self.encodings,
            reverse_encodings=self.reverse_encodings,
            debug=self.debug
        )
        self.meta_manager = MetaGeneManager(
            encodings=self.encodings,
            meta_genes=self.meta_genes,
            meta_gene_stack=self.meta_gene_stack,
            metagene_usage=self.metagene_usage,
            deletion_basket=self.deletion_basket,
            lru_cache_size=lru_cache_size,
            debug=self.debug
        )
        # Explicitly share the unused_encodings reference with meta_manager
        self.meta_manager.unused_encodings = self.unused_encodings
        
        self.organism_generator = OrganismGenerator(
            gene_manager=self.gene_manager,
            reverse_encodings=self.reverse_encodings,
            debug=self.debug
        )

        # Create default delimiters
        # We specifically skip the user-defined approach and forcibly create "Start"=1, "End"=2
        self.add_gene("Start", predefined_id=1)
        self.add_gene("End", predefined_id=2)

    @property
    def gene_counter(self):
        """
        Return the current integer gene counter value.
        This reflects the next ID to be assigned to a new gene
        (or meta-gene) under the hood.

        :return: int, the current gene counter
        """
        return self.gene_counter_ref[0]

    @property
    def lru_cache_size(self):
        """
        Return the LRU cache size used by the MetaGeneManager.

        :return: int, the maximum number of items in the LRU usage tracking
        """
        return self.meta_manager.lru_cache_size

    def generate_hash_key(self, identifier):
        """
        Generate a 64-bit hash key from an identifier.

        :param identifier: The identifier to hash (int or str).
        :return: 64-bit int.
        """
        return xxhash.xxh64_intdigest(str(identifier))

    # -------------------------------------------------------------------------
    # Delegating to GeneManager
    # -------------------------------------------------------------------------
    def add_gene(self, gene, verbose=False, predefined_id=None):
        """
        Add a base gene via the GeneManager.
        """
        return self.gene_manager.add_gene(
            gene=gene,
            verbose=verbose,
            predefined_id=predefined_id,
            generate_hash_key_func=self.generate_hash_key,
            unused_encodings=self.unused_encodings,
            gene_counter_ref=self.gene_counter_ref
        )

    def encode(self, genes, verbose=False):
        """
        Encode a list of gene strings into hash keys.
        """
        return self.gene_manager.encode_genes(genes, verbose=verbose)

    def decode(self, encoded_data, verbose=False):
        """
        Decode an encoded sequence of hash keys into a list of gene strings.
        If verbose is True, we just print debug info (currently unused).
        """

        def update_usage(hash_key):
            # For meta genes, we must update usage so they're not put in the deletion basket
            self.meta_manager.update_metagene_usage(hash_key)

        return self.gene_manager.decode_genes(encoded_data, update_usage_func=update_usage)

    # -------------------------------------------------------------------------
    # Delegating to MetaGeneManager
    # -------------------------------------------------------------------------
    def start_new_generation(self):
        """
        Advance the generation in the meta_manager and handle LRU + deletion.
        """
        # Make sure the meta_manager has the current reference to unused_encodings
        self.meta_manager.unused_encodings = self.unused_encodings
        self.meta_manager.start_new_generation()

    def get_metagene_status(self):
        """
        Return info about current metagene usage, from the meta_manager.
        """
        return self.meta_manager.get_metagene_status()

    def capture_metagene(self, encoded_segment, verbose=False):
        """
        Attempt to capture a new metagene from the provided segment.
        """
        return self.meta_manager.capture_metagene(
            encoded_segment=encoded_segment,
            generate_hash_key_func=self.generate_hash_key,
            unused_encodings=self.unused_encodings,
            gene_counter_ref=self.gene_counter_ref,
            logger=self.logger,
            verbose=verbose
        )

    def open_metagene(self, hash_key, no_delimit=False, verbose=False):
        """
        'Open' a metagene, i.e. decompress it with optional Start/End delimiters.
        """
        start_key = self.reverse_encodings.get("Start")
        end_key = self.reverse_encodings.get("End")
        return self.meta_manager.open_metagene(hash_key, start_key, end_key, no_delimit=no_delimit)

    def delete_metagene(self, hash_key):
        """
        Expose meta_manager's delete method for external calls.
        """
        self.meta_manager.delete_metagene(hash_key)

    # -------------------------------------------------------------------------
    # Delegating to OrganismGenerator
    # -------------------------------------------------------------------------
    def generate_random_organism(self, functional_length=100, include_specials=False,
                                 special_spacing=10, probability=0.99, verbose=False):
        """
        Generate a random organism by selecting from the base gene pool
        plus optional Start/End delimiters.
        """
        return self.organism_generator.generate_random_organism(
            functional_length=functional_length,
            include_specials=include_specials,
            special_spacing=special_spacing,
            probability=probability
        )

    # -------------------------------------------------------------------------
    # Additional / Integrated logic
    # -------------------------------------------------------------------------
    def integrate_uploaded_encodings(self, uploaded_encodings, base_genes, verbose=False):
        """
        Integrate an externally provided encodings structure.
        This method can add new base genes or new metagenes as needed.

        :param uploaded_encodings: A dict or comma-separated string "key:value"
        :param base_genes: The list of base gene strings that are allowed.
        :param verbose: If True, print debug info.
        """
        if isinstance(uploaded_encodings, str):
            uploaded_encodings = {int(k): v for k, v in (item.split(':') for item in uploaded_encodings.split(','))}

        start_key = self.reverse_encodings.get('Start')
        end_key = self.reverse_encodings.get('End')

        for key, value in uploaded_encodings.items():
            if isinstance(value, str):
                # Possibly a base gene
                if value in base_genes or key in [start_key, end_key]:
                    # If we haven't seen it yet, store it
                    if value not in self.reverse_encodings or key in [start_key, end_key]:
                        self.encodings[key] = value
                        self.reverse_encodings[value] = key
            elif isinstance(value, tuple):
                # Then it's a metagene
                self.encodings[key] = value
                self.meta_manager.add_meta_gene(key)
                self.meta_manager.metagene_usage[key] = True

        max_hash_key = max(self.encodings.keys(), default=0)
        if max_hash_key >= self.gene_counter_ref[0]:
            self.gene_counter_ref[0] = max_hash_key + 1

        if verbose and self.debug:
            print(
                f"[EncodingManager] Integrated external encodings. gene_counter_ref is now {self.gene_counter_ref[0]}.")