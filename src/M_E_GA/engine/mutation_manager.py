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
        Mutate an organism by applying various mutation operations based on a
        "stacked probability" model.

        Iterates through the organism. At each index, it builds a list of
        context-appropriate mutations and their independent probabilities.
        A single random roll determines which mutation (if any) is applied.

        :param organism: The encoded organism to mutate.
        :param generation: The current generation number (for logging).
        :param mutation: (Unused) for future extended logic or forced mutation type.
        :param log_enhanced: If True, returns a list of detailed logs.
        :return: The mutated organism. If log_enhanced=True, returns (mutated_organism, logs).
        """
        # Log the "before_mutation" state
        if self.ga.logging and not log_enhanced:
            self.ga.logging_manager.log_organism_state("before_mutation", organism, generation)

        i = 0
        detailed_logs = []

        # Get codons once
        start_codon = self.ga.encoding_manager.reverse_encodings['Start']
        end_codon = self.ga.encoding_manager.reverse_encodings['End']
        delimiter_codons = {start_codon, end_codon}

        # Pre-compute depth at each position in O(n) instead of O(n) per position
        depths = self._compute_depths(organism, start_codon, end_codon)

        while i < len(organism):
            gene = organism[i]
            depth = depths[i]
            
            is_delimiter = gene in delimiter_codons
            
            # --- FIX: Replaced self.ga.encoding_manager.is_metagene(gene) ---
            is_metagene = gene in self.ga.encoding_manager.meta_genes
            # ---------------------------------------------------------------

            # --- 1. Build list of possible mutations and their probabilities ---
            # This replaces the old `select_mutation_type` logic
            candidates = []  # List of (probability, action_name)

            if depth > 0:
                # --- We are INSIDE delimiters ---
                if is_delimiter:
                    candidates.append((self.ga.delimit_delete_prob, 'delimit_delete'))
                else:  # Not a delimiter
                    # Basic mutations (e.g., point, swap, etc.)
                    candidates.append((self.ga.delimited_mutation_prob, 'basic_delimited_package'))
                    # Specialized mutations
                    candidates.append((self.ga.metagene_mutation_prob, 'capture'))
                    candidates.append((self.ga.open_mutation_prob, 'open_no_delimit'))
            
            else:
                # --- We are OUTSIDE delimiters (depth == 0) ---
                if is_delimiter:
                    candidates.append((self.ga.delimit_delete_prob, 'delimit_delete'))
                
                elif is_metagene:
                    candidates.append((self.ga.open_mutation_prob, 'open'))
                    # Also allow basic mutations to *replace* a metagene
                    candidates.append((self.ga.mutation_prob, 'basic_mutation_package'))
                
                else:  # Not delimiter, not metagene
                    # Basic mutations
                    candidates.append((self.ga.mutation_prob, 'basic_mutation_package'))
                    # Specialized mutations
                    candidates.append((self.ga.delimiter_insert_prob, 'insert_delimiter_pair'))

            # --- 2. Perform the "Single Roll" (Roulette Wheel) ---
            roll = random.random()
            prob_sum = 0.0
            action_performed = False

            for prob, action_name in candidates:
                prob_sum += prob
                if roll < prob_sum:
                    # This "slice" was hit

                    # --- 3. Resolve the action ---
                    mutation_type_to_apply = None

                    if action_name == 'basic_mutation_package':
                        # Nested choice for basic mutations (preserves old behavior)
                        basic_choices = ['point', 'swap', 'insertion', 'deletion']
                        mutation_type_to_apply = random.choice(basic_choices)
                    
                    elif action_name == 'basic_delimited_package':
                        # Nested choice for basic mutations (preserves old behavior)
                        basic_choices = ['point', 'swap', 'insertion', 'deletion']
                        mutation_type_to_apply = random.choice(basic_choices)
                    
                    else:
                        # It's a direct action
                        mutation_type_to_apply = action_name

                    # --- 4. Apply the *one* chosen mutation ---
                    if mutation_type_to_apply:
                        # Capture log state *before* applying
                        original_log_state = organism[:] if log_enhanced else None

                        organism, i, mutation_event = self.apply_mutation(
                            organism, i, mutation_type_to_apply, generation
                        )
                        depths = self._compute_depths(organism, start_codon, end_codon)

                        if log_enhanced and mutation_event:
                            detailed_logs.append({
                                "generation": generation,
                                "type": mutation_type_to_apply,
                                "before": original_log_state,
                                "after": organism[:],
                                "index": i,  # `apply_mutation` returns the new, correct index
                                "mutation_event": mutation_event
                            })

                    action_performed = True
                    break  # IMPORTANT: Ensure only one mutation happens per index

            # --- 5. Handle "No Mutation" slice ---
            if not action_performed:
                i += 1  # No mutation, just advance the index

        if log_enhanced:
            return organism, detailed_logs
        return organism

    def apply_mutation(self, organism, index, mutation_type, generation):
        """
        Apply the selected mutation operation on the organism at the given index.

        This method is now called *after* the mutation type has been
        definitively selected by the stacked probability roll.

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
            # No recognized mutation / safety fallback
            index += 1
            return organism, index, None

    @staticmethod
    def _compute_depths(organism, start_codon, end_codon):
        depths = [0] * len(organism)
        depth = 0
        for i, codon in enumerate(organism):
            depths[i] = depth
            if codon == start_codon:
                depth += 1
            elif codon == end_codon and depth > 0:
                depth -= 1
        return depths

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
        # Iterate up to (but not including) the current index
        for i in range(index):
            codon = organism[i]
            if codon == start_codon:
                depth += 1
            elif codon == end_codon:
                # This logic ensures depth can't go negative
                # if the organism is malformed (e.g., "End" "Start")
                if depth > 0:
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
            if weight_sum == 0 or total_meta == 0:
                # Fallback if weights are 0 or no metagenes
                return self.select_gene() 
            
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
        
        # Convert to list for mutation
        organism_list = list(organism)
        
        while i < len(organism_list):
            if organism_list[i] == start_codon:
                # push the index of the Start to the stack
                stack.append(i)
                i += 1
            elif organism_list[i] == end_codon:
                if stack:
                    # pop a matching Start, so we have a valid pair
                    stack.pop()
                    i += 1
                else:
                    # unmatched End, remove it
                    del organism_list[i]
            else:
                i += 1

        # Now remove any leftover unmatched Start(s)
        # They are at indices in stack, which might be out-of-date if we've deleted codons in the loop
        # So we do it carefully from the end.
        for idx in reversed(stack):
            del organism_list[idx]

        return tuple(organism_list)

    def log_mutation_if_needed(self, mutation_log):
        """
        Helper method to log the mutation event if logging is enabled.

        :param mutation_log: Dictionary containing mutation details.
        :return: None
        """
        if self.ga.logging and self.ga.mutation_logging:
            self.ga.logging_manager.log_mutation(mutation_log)