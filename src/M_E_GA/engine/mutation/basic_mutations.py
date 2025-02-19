"""
basic_mutations.py

Contains functions for basic mutation operations like insertion,
point mutation, swap, and deletion.

All these functions follow SRP and do not store any state themselves.
They simply perform an action on the organism and return the new state.
"""

import random


def perform_insertion(organism, index, generation, manager):
    """
    Insert a new gene at the specified index.

    :param organism: The encoded organism (list of hash keys).
    :param index: The position where we'll insert the new gene.
    :param generation: Current generation number (for logging).
    :param manager: The MutationManager instance for accessing GA config/logging.
    :return: (organism, new_index, mutation_log).
    """
    ga = manager.ga
    gene_key = manager.select_gene()
    gene_name = ga.encoding_manager.encodings.get(gene_key, "Unknown")
    organism.insert(index, gene_key)

    mutation_log = {
        'type': 'insertion',
        'generation': generation,
        'index': index,
        'gene_inserted': gene_name,
        'codon_inserted': gene_key
    }
    manager.log_mutation_if_needed(mutation_log)

    return organism, index + 1, mutation_log


def perform_point_mutation(organism, index, generation, manager):
    """
    Replace the gene at 'index' with a newly selected gene.

    :param organism: The encoded organism.
    :param index: The index in the organism to mutate.
    :param generation: Current generation number.
    :param manager: The MutationManager instance for referencing GA/logging.
    :return: (organism, new_index, mutation_log).
    """
    ga = manager.ga
    original_codon = organism[index]
    new_codon = manager.select_gene()
    organism[index] = new_codon

    gene_name = ga.encoding_manager.encodings.get(new_codon, "Unknown")
    mutation_log = {
        'type': 'point_mutation',
        'generation': generation,
        'index': index,
        'original_codon': original_codon,
        'new_codon': new_codon,
        'gene': gene_name
    }
    manager.log_mutation_if_needed(mutation_log)

    # We don't advance the index because we replaced in place
    return organism, index, mutation_log


def perform_swap(organism, index, generation, manager):
    """
    Swap the gene at 'index' with one of its neighbors.

    :param organism: The encoded organism.
    :param index: The position in the organism to attempt a swap.
    :param generation: Current generation number.
    :param manager: The MutationManager instance for referencing GA/logging.
    :return: (organism, new_index, mutation_log).
    """
    ga = manager.ga

    swap_actions = ['forward', 'backward']
    first_action = random.choice(swap_actions)
    swapped = False
    swapped_index = index

    if first_action == 'forward' and manager.can_swap(organism, index, index + 1):
        organism[index], organism[index + 1] = organism[index + 1], organism[index]
        swapped_index = index + 1
        swapped = True
    elif manager.can_swap(organism, index, index - 1):
        organism[index], organism[index - 1] = organism[index - 1], organism[index]
        swapped_index = index - 1
        swapped = True

    mutation_log = None
    if swapped:
        mutation_log = {
            'type': 'swap',
            'generation': generation,
            'index': index,
            'swapped_with_index': swapped_index,
            'codon_at_new_index': organism[swapped_index],
            'codon_swapped': organism[index]
        }
        manager.log_mutation_if_needed(mutation_log)

    # We don't move index if we didn't do a valid swap, so we just keep iterating
    return organism, index, mutation_log


def perform_deletion(organism, index, generation, manager):
    """
    Delete a single gene at the specified index (unless the organism is only length 1).

    :param organism: The encoded organism.
    :param index: The position in the organism from which to delete a gene.
    :param generation: Current generation number.
    :param manager: The MutationManager instance for referencing GA/logging.
    :return: (organism, new_index, mutation_log).
    """
    if len(organism) > 1:
        deleted_codon = organism[index]
        del organism[index]

        mutation_log = {
            'type': 'deletion',
            'generation': generation,
            'index': index,
            'deleted_codon': deleted_codon
        }
        manager.log_mutation_if_needed(mutation_log)
        index = max(0, index - 1)
        return organism, index, mutation_log

    return organism, index + 1, None
