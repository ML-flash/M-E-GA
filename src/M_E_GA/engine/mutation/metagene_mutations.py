"""
metagene_mutations.py

Handles capturing a delimited segment as a metagene,
and opening (expanding) a metagene back into its content.
"""


def perform_capture(organism, index, generation, manager):
    """
    Compress a delimited segment into a metagene.

    :param organism: The encoded organism.
    :param index: Position from which to find a delimiter pair.
    :param generation: Current generation.
    :param manager: The MutationManager instance for referencing GA/logging.
    :return: (organism, new_index, mutation_log).
    """
    ga = manager.ga
    delimiter_pair = manager.find_delimiters(organism, index)
    if delimiter_pair is not None:
        start_idx, end_idx = delimiter_pair
        segment_size = end_idx - start_idx - 1
        if segment_size > 1:
            segment_to_capture = organism[start_idx + 1:end_idx]
            captured_codon = ga.encoding_manager.capture_metagene(segment_to_capture)
            if captured_codon is not False:
                organism = organism[:start_idx] + [captured_codon] + organism[end_idx + 1:]
                mutation_log = {
                    'type': 'capture',
                    'generation': generation,
                    'index': start_idx,
                    'captured_segment': segment_to_capture,
                    'captured_codon': captured_codon
                }
                manager.log_mutation_if_needed(mutation_log)
                return organism, start_idx, mutation_log

    return organism, index + 1, None


def perform_open(organism, index, generation, manager, no_delimit=False):
    """
    Expand a metagene back into its content, optionally with or without delimiters.

    :param organism: The encoded organism.
    :param index: Position in the organism.
    :param generation: Current generation.
    :param manager: The MutationManager instance for referencing GA/logging.
    :param no_delimit: If True, do not add Start/End around the decompressed content.
    :return: (organism, new_index, mutation_log).
    """
    ga = manager.ga
    decompressed = ga.encoding_manager.open_metagene(organism[index], no_delimit=no_delimit)
    if decompressed is not False:
        organism = organism[:index] + decompressed + organism[index + 1:]
        # Adjust index to skip the expanded contents
        new_index = index + len(decompressed) - 1
        mutation_log = {
            'type': 'open' if not no_delimit else 'open_no_delimit',
            'generation': generation,
            'index': new_index,
            'decompressed_content': decompressed
        }
        manager.log_mutation_if_needed(mutation_log)
        return organism, new_index, mutation_log

    return organism, index + 1, None
