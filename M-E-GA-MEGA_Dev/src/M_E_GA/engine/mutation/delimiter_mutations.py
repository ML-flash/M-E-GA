"""
delimiter_mutations.py

Handles mutations specifically related to delimiter codons:
- Removing delimited segments
- Inserting new Start/End pairs
"""


def perform_delimit_delete(organism, index, generation, manager):
    """
    Delete a segment enclosed by delimiters if found at or before `index`.

    :param organism: The encoded organism.
    :param index: The position from which we try to find a delimiter pair.
    :param generation: Current generation number.
    :param manager: The MutationManager instance for referencing GA/logging.
    :return: (organism, new_index, mutation_log).
    """
    ga = manager.ga
    delimiter_pair = manager.find_delimiters(organism, index)
    if delimiter_pair is not None:
        start_loc, end_loc = delimiter_pair
        if start_loc + 1 < end_loc:
            # slice out the content from start_loc..end_loc
            organism = organism[:start_loc] + organism[start_loc + 1:end_loc] + organism[end_loc + 1:]
        else:
            # just remove Start and End if no content in between
            organism = organism[:start_loc] + organism[end_loc + 1:]

        mutation_log = {
            'type': 'delimit_delete',
            'generation': generation,
            'start_location': start_loc,
            'end_location': end_loc
        }
        manager.log_mutation_if_needed(mutation_log)
        # Move index back to start_loc
        index = start_loc
        return organism, index, mutation_log

    # If we can't find a pair, do nothing
    return organism, index + 1, None


def insert_delimiter_pair(organism, index, generation, manager):
    """
    Insert a 'Start' and 'End' pair at the current index or near it.

    :param organism: The encoded organism.
    :param index: The position at which we insert the Start codon (the End is after).
    :param generation: Current generation number.
    :param manager: The MutationManager instance for referencing GA/logging.
    :return: (organism, new_index, mutation_log).
    """
    ga = manager.ga
    start_codon = ga.encoding_manager.reverse_encodings['Start']
    end_codon = ga.encoding_manager.reverse_encodings['End']

    organism.insert(index, start_codon)
    end_delimiter_index = index + 2
    if end_delimiter_index <= len(organism):
        organism.insert(end_delimiter_index, end_codon)
    else:
        organism.append(end_codon)

    mutation_log = {
        'type': 'insert_delimiter_pair',
        'generation': generation,
        'index': index
    }
    manager.log_mutation_if_needed(mutation_log)
    return organism, end_delimiter_index, mutation_log
