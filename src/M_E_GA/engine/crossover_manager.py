"""
crossover_manager.py

Responsible for handling crossover logic such as choosing the crossover point,
swapping segments, and skipping delimited sections if necessary.
"""

import random


class CrossoverManager:
    """
    Handles the crossover operations for two parent organisms,
    including logic to avoid messing with delimited sections.
    """

    def __init__(self, ga_instance):
        """
        Initialize the CrossoverManager.

        :param ga_instance: Reference to the main GA instance to access config, logs, etc.
        """
        self.ga = ga_instance

    def crossover(self, parent1, parent2, non_delimited_indices, generation):
        """
        Perform a crossover operation between two parents at a random non-delimited index.

        :param parent1: The first parent's organism encoding.
        :param parent2: The second parent's organism encoding.
        :param non_delimited_indices: Indices that are safe for crossover (no Start/End messing).
        :param generation: Current generation number (for logging).
        :return: (offspring1, offspring2) after the crossover.
        """
        crossover_point = self.choose_crossover_point(non_delimited_indices)
        if crossover_point is None:
            offspring1, offspring2 = parent1[:], parent2[:]
        else:
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

        # Updated reference to the logging manager
        self.ga.logging_manager.log_crossover(
            generation, parent1, parent2, crossover_point, offspring1, offspring2
        )
        return offspring1, offspring2

    def choose_crossover_point(self, non_delimited_indices):
        """
        Randomly select a crossover point from the provided non-delimited indices.

        :param non_delimited_indices: A list of valid indices for crossover.
        :return: The selected crossover index or None if no valid index exists.
        """
        if not non_delimited_indices:
            return None
        return random.choice(non_delimited_indices)

    def get_non_delimiter_indices(self, parent1, parent2):
        """
        Determine the indices that are not part of delimited segments for both parents.

        :param parent1: First parent's organism encoding.
        :param parent2: Second parent's organism encoding.
        :return: A list of indices that are safe to use for crossover.
        """
        delimiter_indices = self.calculate_delimiter_indices(parent1, parent2)
        min_len = min(len(parent1), len(parent2))
        non_delimited = set(range(min_len))

        for (start_idx, end_idx) in delimiter_indices:
            # Remove those indices from the available set
            for i in range(start_idx, end_idx + 1):
                if i in non_delimited:
                    non_delimited.remove(i)

        return list(non_delimited)

    def calculate_delimiter_indices(self, parent1, parent2):
        """
        Calculate the indices of delimiter segments for both parents combined.

        :param parent1: The first parent's organism encoding.
        :param parent2: The second parent's organism encoding.
        :return: A list of (start, end) index pairs indicating delimited ranges.
        """
        # We'll gather ranges for each parent
        combined = []
        combined.extend(self._extract_delimiter_ranges(parent1))
        combined.extend(self._extract_delimiter_ranges(parent2))
        return combined

    def _extract_delimiter_ranges(self, organism):
        """
        Helper to parse an organism and find all (start, end) delimiter pairs in the encoding.
        """
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']
        starts = [i for i, codon in enumerate(organism) if codon == start_codon]
        ends = [i for i, codon in enumerate(organism) if codon == end_codon]
        return list(zip(starts, ends))

    def is_fully_delimited(self, organism):
        """
        Check if an organism is fully delimited: i.e. if it starts with 'Start'
        and ends with 'End' and doesn't contain more content outside.

        :param organism: The encoded organism list.
        :return: True if fully delimited, otherwise False.
        """
        if not organism:
            return False
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']
        return (organism[0] == start_codon) and (organism[-1] == end_codon)
