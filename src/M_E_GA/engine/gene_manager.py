"""
gene_manager.py

Handles base gene operations such as adding genes, encoding gene strings,
and decoding encoded gene sequences.

Part of the refactoring for Single Responsibility Principle.
"""

import functools


class GeneManager:
    """
    Manages the addition of base genes and the encoding/decoding
    of gene sequences.

    Attributes:
        encodings (dict): Maps integer hash keys to either base gene strings or metagene tuples.
        reverse_encodings (dict): Maps gene strings to their integer hash keys.
        debug (bool): Flag to enable or disable verbose debugging output.
    """

    def __init__(self, encodings, reverse_encodings, debug=False):
        """
        Initialize the GeneManager.

        :param encodings: Reference to a dictionary storing hash -> gene/metagene content.
        :param reverse_encodings: Reference to a dictionary storing gene string -> hash.
        :param debug: If True, enables debug print statements.
        """
        self.encodings = encodings
        self.reverse_encodings = reverse_encodings
        self.debug = debug

    def add_gene(self, gene, verbose=False, predefined_id=None, generate_hash_key_func=None, unused_encodings=None,
                 gene_counter_ref=None):
        """
        Adds a new gene to the encodings unless it already exists. Returns its hash key.

        :param gene: The gene string to add.
        :param verbose: If True, prints additional information.
        :param predefined_id: Optional int used to force a specific id for hashing.
        :param generate_hash_key_func: A function that creates a 64-bit integer hash from an identifier.
        :param unused_encodings: A list from which we can reuse freed hash keys if any exist.
        :param gene_counter_ref: A mutable integer reference used for assigning new IDs.
        :return: The integer hash key corresponding to the gene.
        """
        if gene in self.reverse_encodings:
            return self.reverse_encodings[gene]

        if not generate_hash_key_func:
            raise ValueError("GeneManager requires a generate_hash_key_func to create new hash keys.")

        if unused_encodings is not None and unused_encodings and predefined_id is None:
            hash_key = unused_encodings.pop(0)
        else:
            if predefined_id is not None:
                identifier = predefined_id
            else:
                if gene_counter_ref is None:
                    raise ValueError("gene_counter_ref is required if no predefined_id is provided.")
                identifier = gene_counter_ref[0]  # gene_counter_ref is a list with one element
                gene_counter_ref[0] += 1

            hash_key = generate_hash_key_func(identifier)

        self.encodings[hash_key] = gene
        self.reverse_encodings[gene] = hash_key

        if verbose and self.debug:
            print(f"[GeneManager] Added gene '{gene}' with hash {hash_key}.")

        return hash_key

    def encode_genes(self, genes, verbose=False):
        """
        Encodes a list of gene strings into their corresponding hash keys.

        :param genes: A list of gene strings to encode.
        :param verbose: If True, prints warning for unrecognized genes.
        :return: A list of integer hash keys.
        """
        encoded_list = []
        for gene in genes:
            if gene not in self.reverse_encodings:
                if verbose and self.debug:
                    print(f"[GeneManager] Unrecognized gene '{gene}'. Skipping encoding.")
                continue
            encoded_list.append(self.reverse_encodings[gene])
        return encoded_list

    @functools.lru_cache(maxsize=1000)
    def decode_genes(self, encoded_tuple, update_usage_func=None):
        """
        Decodes an encoded tuple of hash keys back into the original gene sequence.
        Utilizes an LRU cache for efficiency.

        :param encoded_tuple: A tuple (or a single int) representing encoded genes/metagenes.
        :param update_usage_func: Callback to update usage record for meta-genes, if needed.
        :return: A list of gene strings, which may also contain "Unknown".
        """
        if not encoded_tuple:
            return []

        if not isinstance(encoded_tuple, tuple):
            encoded_tuple = (encoded_tuple,)

        stack = list(encoded_tuple)
        decoded_sequence = []

        while stack:
            hash_key = stack.pop(0)
            if hash_key in self.encodings:
                value = self.encodings[hash_key]
                # If there's a meta-usage function, update usage.
                if update_usage_func:
                    update_usage_func(hash_key)

                # If it's a tuple, we expand it
                if isinstance(value, tuple):
                    stack = list(value) + stack
                else:
                    decoded_sequence.append(value)
            else:
                decoded_sequence.append("Unknown")

        return decoded_sequence
