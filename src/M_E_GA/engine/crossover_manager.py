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

    def crossover(self, parent1, parent2, generation):
        """
        Perform a crossover operation between two parents at independent depth-zero positions.
        
        Each parent is cut at a randomly selected depth-zero position (outside all delimiter pairs).
        The cut points do NOT need to align between parents.

        :param parent1: The first parent's organism encoding.
        :param parent2: The second parent's organism encoding.
        :param generation: Current generation number (for logging).
        :return: (offspring1, offspring2) after the crossover.
        """
        valid_cuts_p1 = self.get_depth_zero_cuts(parent1)
        valid_cuts_p2 = self.get_depth_zero_cuts(parent2)
        
        if not valid_cuts_p1 or not valid_cuts_p2:
            # No valid cuts in one or both parents - return copies
            offspring1, offspring2 = parent1[:], parent2[:]
            crossover_point = None
        else:
            # Independent random cuts for each parent
            cut1 = random.choice(valid_cuts_p1)
            cut2 = random.choice(valid_cuts_p2)
            
            # Swap segments after cut points
            offspring1 = parent1[:cut1] + parent2[cut2:]
            offspring2 = parent2[:cut2] + parent1[cut1:]
            
            crossover_point = (cut1, cut2)

        # Log the crossover event
        self.ga.logging_manager.log_crossover(
            generation, parent1, parent2, crossover_point, offspring1, offspring2
        )
        return offspring1, offspring2

    def get_depth_zero_cuts(self, parent):
        """
        Get all valid crossover cut positions for a parent.
        A valid cut position is after any index where delimiter depth is zero.

        :param parent: The parent's organism encoding.
        :return: List of valid cut positions (indices after which to cut).
        """
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']
        
        valid_cuts = []
        depth = 0
        
        for i in range(len(parent)):
            # Update depth based on current token
            if parent[i] == start_codon:
                depth += 1
            elif parent[i] == end_codon:
                depth -= 1
            
            # Valid cut: after this position, if depth is zero
            # Don't cut after the last element (would produce empty offspring)
            if depth == 0 and i < len(parent) - 1:
                valid_cuts.append(i + 1)
        
        return valid_cuts

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