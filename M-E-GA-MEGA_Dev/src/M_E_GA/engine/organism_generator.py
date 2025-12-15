"""
organism_generator.py

Generates random organisms from the existing gene pool
while optionally injecting special delimiters.

Part of the refactoring for Single Responsibility Principle.
"""

import random


class OrganismGenerator:
    """
    Handles the creation of random organisms from base genes (and
    potentially Start/End delimiters).
    """

    def __init__(self, gene_manager, reverse_encodings, debug=False):
        """
        Initialize the OrganismGenerator.

        :param gene_manager: Instance of GeneManager to help with encoding.
        :param reverse_encodings: The global reverse encodings dict from gene -> hash.
        :param debug: Whether to enable debug logs.
        """
        self.gene_manager = gene_manager
        self.reverse_encodings = reverse_encodings
        self.debug = debug

    def generate_random_organism(self, functional_length=100, include_specials=False,
                                 special_spacing=10, probability=0.99):
        """
        Generates a random organism by selecting random genes, optionally
        inserting 'Start'/'End' delimiter pairs at intervals.

        :param functional_length: The number of functional genes to include.
        :param include_specials: If True, allow insertion of Start/End pairs.
        :param special_spacing: The minimal spacing between delimiter pairs.
        :param probability: Probability threshold for deciding to insert a pair.
        :return: The encoded organism as a list of hash keys.
        """
        # Exclude Start/End from the base gene pool
        start_hash = self.reverse_encodings.get("Start")
        end_hash = self.reverse_encodings.get("End")
        gene_pool = [gene for gene in self.reverse_encodings if gene not in ["Start", "End"]]

        # Build a random list of base genes
        organism_genes = [random.choice(gene_pool) for _ in range(functional_length)]

        special_gene_indices = set()

        if include_specials and start_hash is not None and end_hash is not None:
            for i in range(len(organism_genes)):
                # Maybe insert a Start/End pair
                if random.random() <= probability:
                    # Check spacing
                    if all(abs(i - idx) >= special_spacing for idx in special_gene_indices):
                        organism_genes.insert(i, "Start")
                        end_index = min(i + special_spacing, len(organism_genes))
                        organism_genes.insert(end_index, "End")
                        special_gene_indices.update([i, end_index])

        # Encode them
        return self.gene_manager.encode_genes(organism_genes, verbose=False)
