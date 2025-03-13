"""
test_metagene_deletion_edge_case.py

This test file is intended to specifically stress-test the meta-gene deletion process,
attempting to replicate or expose any edge cases that cause leftover "Unknown" references.

We'll:
1. Create a GA with a relatively large population.
2. Use random capturing, nested references, and generational passes to see if
   any meta-genes get incorrectly inlined or lead to "Unknown" decodes.
3. Decode in strict mode to ensure we catch unknown references as soon as they appear.

If we pass without exceptions, presumably the deletion process is stable under these stresses.
"""

import unittest
import random
from src.M_E_GA import M_E_GA_Base


def random_fitness_function(encoded_org, ga):
    """
    A trivial fitness function that just returns a random float.
    We don't care about actual fitness here; we only want to push the GA through many generations.
    """
    return random.random()


class TestMetaGeneDeletionEdgeCase(unittest.TestCase):
    """
    A dedicated test suite that repeatedly triggers meta-gene creation and deletion, verifying no "Unknown" arises.
    """

    def test_meta_gene_deletion_stressful_runs(self):
        """
        Stress test for meta-gene deletion:
        - We create a large population.
        - We set mutation probabilities to encourage meta-gene usage and delimiter usage.
        - We run multiple generations in 'strict_decode' mode, so if any unknown references exist, we get an error.
        - If no error is raised, it implies we didn't leave behind partial references after deletion.
        """

        # Make our GA instance:
        # We'll choose somewhat large population and enable more frequent meta-gene capturing and delimiters.
        ga = M_E_GA_Base(
            genes=['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE'],
            fitness_function=random_fitness_function,
            population_size=1000,       # large enough to stress but not too slow
            max_individual_length=15,   # allow some decent size
            num_parents=20,
            max_generations=200,        # do 200 generations
            mutation_prob=0.1,          # overall mutation chance
            delimited_mutation_prob=0.1,
            metagene_mutation_prob=0.1, # encourage meta-gene capturing
            open_mutation_prob=0.05,    # chance of opening meta-genes
            delimiter_insert_prob=0.05, # chance of inserting new delimiters
            strict_decode=True,         # *strict* decode to catch unknown references
            logging=False               # keep logs off to speed up test
        )

        # Initialize population
        population = ga.initialize_population()

        # Actually run the GA. If meta-gene deletion produces any "Unknown" references, the run fails.
        # The run_algorithm method calls 'start_new_generation' which triggers metagene cleanup.
        ga.run_algorithm()

        # If we get here, no unknown references were found
        # Let's do a final strict decode pass on the final population (redundant but let's be sure).
        for org_idx, organism in enumerate(ga.population):
            ga.encoding_manager.gene_manager.decode_genes(
                tuple(organism),
                update_usage_func=ga.encoding_manager.meta_manager.update_metagene_usage,
                raise_on_unknown=True,
                decode_context=f"Final check of organism {org_idx}"
            )

        # If we haven't crashed, assume the process is stable enough.
        self.assertTrue(True, "Stress test completed with no unknown references found.")


if __name__ == '__main__':
    unittest.main()
