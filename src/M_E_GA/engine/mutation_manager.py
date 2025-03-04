"""
mutation_manager.py

Handles high-level mutation orchestration, delegating specific mutation operations
to separate modules under src/M_E_GA/engine/mutation/.
"""

import random

# Import specialized mutation functions
from .mutation.basic_mutations import (
    perform_insertion,
    perform_point_mutation,
    perform_swap,
    perform_deletion
)
from .mutation.delimiter_mutations import (
    perform_delimit_delete,
    insert_delimiter_pair
)
from .mutation.metagene_mutations import (
    perform_capture,
    perform_open
)


class MutationManager:
    """
    MutationManager is responsible for orchestrating organism-level mutation logic.
    It delegates specific mutations (insertion, deletion, swap, capture, etc.)
    to smaller modules that follow SRP more closely.
    """

    def __init__(self, ga_instance):
        """
        Initialize the MutationManager.

        :param ga_instance: The main M_E_GA_Base instance for accessing config, logs, etc.
        """
        self.ga = ga_instance

    def mutate_organism(self, organism, generation, mutation=None, log_enhanced=False):
        """
        Mutate an organism by applying various mutation operations.

        Iterates through the organism and applies a mutation based on probabilities,
        type, and depth (inside or outside delimiters).

        :param organism: The encoded organism to mutate.
        :param generation: The current generation number (for logging).
        :param mutation: (Unused) for future extended logic or forced mutation type.
        :param log_enhanced: If True, returns a list of detailed logs (unused by default).
        :return: The mutated organism. If log_enhanced=True, returns (mutated_organism, logs).
        """
        # Log the "before_mutation" state
        if self.ga.logging and not log_enhanced:
            self.ga.logging_manager.log_organism_state("before_mutation", organism, generation)

        i = 0
        detailed_logs = []

        while i < len(organism):
            original = organism[:]
            depth = self.calculate_depth(organism, i)

            # Are we inside delimiters?
            if depth > 0:
                mutation_prob = self.ga.delimited_mutation_prob
            else:
                mutation_prob = self.ga.mutation_prob

            if random.random() <= mutation_prob:
                mutation_type = self.select_mutation_type(organism, i, depth)
                organism, i, mutation_event = self.apply_mutation(organism, i, mutation_type, generation)
                if log_enhanced and mutation_event is not None:
                    detailed_logs.append({
                        "generation": generation,
                        "type": mutation_type,
                        "before": original,
                        "after": organism[:],
                        "index": i,
                        "mutation_event": mutation_event
                    })
            else:
                i += 1

        if log_enhanced:
            return organism, detailed_logs
        return organism

    def select_mutation_type(self, organism, index, depth):
        """
        Select the type of mutation to perform based on the gene and its context.

        :param organism: The encoded organism list.
        :param index: The current index in the organism.
        :param depth: The nesting depth (inside delimiters?).
        :return: The string representing the chosen mutation type.
        """
        gene = organism[index]
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']

        # If it's a Start/End codon
        if gene in {start_codon, end_codon}:
            # Possibly a delimiter deletion or swap
            if random.random() < self.ga.delimit_delete_prob:
                return 'delimit_delete'
            else:
                return 'swap'
        else:
            # If inside delimiters
            if depth > 0:
                mutation_choices = [
                    'point', 'swap', 'insertion', 'deletion',
                    'capture', 'open_no_delimit'
                ]
                mutation_weights = [
                    1.0,  # point
                    1.0,  # swap
                    1.0,  # insertion
                    1.0,  # deletion
                    self.ga.metagene_mutation_prob,  # capture
                    self.ga.open_mutation_prob  # open_no_delimit
                ]
            else:
                mutation_choices = [
                    'point', 'swap', 'insertion', 'deletion',
                    'insert_delimiter_pair', 'open'
                ]
                mutation_weights = [
                    1.0,  # point
                    1.0,  # swap
                    1.0,  # insertion
                    1.0,  # deletion
                    self.ga.delimiter_insert_prob,  # insert_delimiter_pair
                    self.ga.open_mutation_prob  # open
                ]

            total_weight = sum(mutation_weights)
            normalized_probs = [w / total_weight for w in mutation_weights]
            mutation_type = random.choices(mutation_choices, weights=normalized_probs, k=1)[0]
            return mutation_type

    def apply_mutation(self, organism, index, mutation_type, generation):
        """
        Apply the selected mutation operation on the organism at the given index.

        :param organism: The encoded organism.
        :param index: Current position in the organism.
        :param mutation_type: The type of mutation to apply.
        :param generation: Current generation number.
        :return: (mutated_organism, new_index, mutation_event) tuple
                 where mutation_event is a dict describing the mutation, or None.
        """
        if mutation_type == 'insertion':
            return perform_insertion(organism, index, generation, self)
        elif mutation_type == 'point':
            return perform_point_mutation(organism, index, generation, self)
        elif mutation_type == 'swap':
            return perform_swap(organism, index, generation, self)
        elif mutation_type == 'delimit_delete':
            return perform_delimit_delete(organism, index, generation, self)
        elif mutation_type == 'deletion':
            return perform_deletion(organism, index, generation, self)
        elif mutation_type == 'capture':
            return perform_capture(organism, index, generation, self)
        elif mutation_type == 'open':
            return perform_open(organism, index, generation, self, no_delimit=False)
        elif mutation_type == 'open_no_delimit':
            return perform_open(organism, index, generation, self, no_delimit=True)
        elif mutation_type == 'insert_delimiter_pair':
            return insert_delimiter_pair(organism, index, generation, self)
        else:
            # No recognized mutation
            index += 1
            return organism, index, None

    def calculate_depth(self, organism, index):
        """
        Calculate the nesting depth at a given index by counting Start/End delimiters.

        :param organism: The encoded organism list.
        :param index: Position in the organism to check.
        :return: An integer representing the current depth.
        """
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']
        depth = 0
        for codon in organism[:index + 1]:
            if codon == start_codon:
                depth += 1
            elif codon == end_codon:
                depth -= 1
        return depth

    def select_gene(self):
        """
        Select a gene for insertion or point mutation, either from the base genes
        or from the meta-gene stack, weighted by self.ga.metagene_prob.

        :return: The selected hash key for that gene.
        """
        import random
        if random.random() < self.ga.base_gene_prob or not self.ga.encoding_manager.meta_genes:
            # pick a base gene
            base_gene = random.choice(self.ga.genes)
            if base_gene not in ['Start', 'End']:
                return self.ga.encoding_manager.reverse_encodings[base_gene]
            else:
                return self.select_gene()  # try again if it's Start/End
        else:
            # pick from meta_genes
            meta_gene_keys = self.ga.encoding_manager.meta_gene_stack
            total_meta = len(meta_gene_keys)
            weights = [
                self.ga.metagene_prob ** (total_meta - i - 1)
                for i in range(total_meta)
            ]
            weight_sum = sum(weights)
            if weight_sum == 0:
                normalized_weights = [1.0 / total_meta] * total_meta
            else:
                normalized_weights = [w / weight_sum for w in weights]
            gene_key = random.choices(meta_gene_keys, weights=normalized_weights, k=1)[0]
            return gene_key

    def can_swap(self, organism, index_a, index_b):
        """
        Check whether two positions in the organism can be swapped (i.e., they exist
        and aren't both delimiters).
        """
        if 0 <= index_a < len(organism) and 0 <= index_b < len(organism):
            start_encoding = self.ga.encoding_manager.reverse_encodings['Start']
            end_encoding = self.ga.encoding_manager.reverse_encodings['End']
            if (organism[index_a] in [start_encoding, end_encoding]
                    and organism[index_b] in [start_encoding, end_encoding]):
                return False
            return True
        return False

    def find_delimiters(self, organism, index):
        """
        Find the closest 'Start' and 'End' codons around 'index'.

        :param organism: The encoded organism.
        :param index: The index from which to search for delimiters.
        :return: (start_index, end_index) or None if not found.
        """
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']
        start_index, end_index = None, None

        # Look backward for the most recent Start
        for i in range(index, -1, -1):
            if organism[i] == start_codon:
                start_index = i
                break

        if start_index is not None:
            # Look forward for the first End
            for j in range(start_index + 1, len(organism)):
                if organism[j] == end_codon:
                    end_index = j
                    break

        if start_index is not None and end_index is not None:
            return (start_index, end_index)
        return None

    def repair(self, organism):
        """
        Repair an organism by ensuring all delimiters are matched.
        Remove every unmatched Start or End codon.
        """
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']

        # We'll store indices of valid items here:
        # or we can build a new list of codons and store them, but we also need to remove unmatched easily
        # Using "stack of indices" approach:
        stack = []
        i = 0
        while i < len(organism):
            if organism[i] == start_codon:
                # push the index of the Start to the stack
                stack.append(i)
                i += 1
            elif organism[i] == end_codon:
                if stack:
                    # pop a matching Start, so we have a valid pair
                    stack.pop()
                    i += 1
                else:
                    # unmatched End, remove it
                    del organism[i]
            else:
                i += 1

        # Now remove any leftover unmatched Start(s)
        # They are at indices in stack, which might be out-of-date if we've deleted codons in the loop
        # So we do it carefully from the end.
        for idx in reversed(stack):
            del organism[idx]

        return organism

    def log_mutation_if_needed(self, mutation_log):
        """
        Helper method to log the mutation event if logging is enabled.

        :param mutation_log: Dictionary containing mutation details.
        :return: None
        """
        if self.ga.logging and self.ga.mutation_logging:
            self.ga.logging_manager.log_mutation(mutation_log)
