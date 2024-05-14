import json
import datetime
import random
from M_E_GA_Base_V2 import M_E_GA_Base
import matplotlib.pyplot as plt
import numpy as np

class Experiment:
    def __init__(self, ga_base, seed=None, mutation_cycles=100):
        self.ga_base = ga_base
        self.mutation_cycles = mutation_cycles
        self.logs = []
        if seed is not None:
            random.seed(seed)
            self.ga_base.seed = seed

    def visualize(self):
        # Determine the maximum lengths for proper scaling in the plot
        max_len_encoded = max(len(org) for org in self.visual_data_encoded)
        max_len_decoded = max(len(org) for org in self.visual_data_decoded)
        max_organism_length = max(max_len_encoded, max_len_decoded)  # Largest size of any organism

        # Initialize the grids with zeros, which will be overwritten based on organism size
        grid_encoded = np.zeros(
            (self.mutation_cycles + 1, max_len_decoded))  # Use max_len_decoded for both to align the organisms
        grid_decoded = np.zeros((self.mutation_cycles + 1, max_len_decoded))

        # Fill out the grids, representing the organisms in white and the background based on size
        for i in range(self.mutation_cycles + 1):
            encoded_organism = self.visual_data_encoded[i]
            decoded_organism = self.visual_data_decoded[i]

            # Calculate the offsets for centering
            offset_encoded = (max_len_decoded - len(encoded_organism)) // 2
            offset_decoded = (max_len_decoded - len(decoded_organism)) // 2

            # Assign a value based on organism size for the background of the row, normalized by the largest organism
            background_encoded_value = len(
                encoded_organism) / max_organism_length  # Normalize based on the largest organism size
            background_decoded_value = len(decoded_organism) / max_organism_length

            # Fill the whole row with the background value
            grid_encoded[i, :] = background_encoded_value
            grid_decoded[i, :] = background_decoded_value

            # Represent organisms in white
            grid_encoded[i, offset_encoded:offset_encoded + len(encoded_organism)] = 1
            grid_decoded[i, offset_decoded:offset_decoded + len(decoded_organism)] = 1

        # Plot the grids
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Encoded organism plot
        cmap = plt.cm.hot
        cmap.set_over('white')  # Ensure that the organism (maximum value) is white
        ax1.imshow(grid_encoded, cmap=cmap, aspect='auto', interpolation='nearest', vmax=1)
        ax1.set_title('Encoded Organism Mutation Visualization')
        ax1.set_xlabel('Gene Position')
        ax1.set_ylabel('Mutation Cycle')

        # Decoded organism plot
        ax2.imshow(grid_decoded, cmap=cmap, aspect='auto', interpolation='nearest', vmax=1)
        ax2.set_title('Decoded Organism Mutation Visualization')
        ax2.set_xlabel('Gene Position')
        ax2.set_ylabel('Mutation Cycle')

        plt.tight_layout()
        plt.show()

    def log_organism(self, organism, step):
        self.logs.append({
            "step": step,
            "encoded": organism[:],
            "decoded": self.ga_base.decode_organism(organism)
        })

    def log_enhanced_mutation(self, before, after, mutation_logs, cycle):
        for mutation in mutation_logs:
            self.logs.append({
                "cycle": cycle,
                "mutation_type": mutation['type'],
                "before": {
                    "encoded": before[:],
                    "decoded": self.ga_base.decode_organism(before)
                },
                "after": {
                    "encoded": after[:],
                    "decoded": self.ga_base.decode_organism(after)
                }
            })

    def print_logs(self):
        for log in self.logs:
            if 'mutation_type' in log:
                print(f"Cycle {log['cycle']} Mutation:")
                print(f"  Type: {log['mutation_type']}")
                print(f"  Before - Encoded: {log['before']['encoded']}")
                print(f"           Decoded: {log['before']['decoded']}")
                print(f"  After - Encoded: {log['after']['encoded']}")
                print(f"          Decoded: {log['after']['decoded']}\n")
            else:
                print(f"Initial Organism - Cycle {log['step']}:")
                print(f"  Encoded: {log['encoded']}")
                print(f"  Decoded: {log['decoded']}\n")
        print("End of Experiment Logging.")

    def run(self):
        self.visual_data_encoded = []
        self.visual_data_decoded = []

        organism = self.ga_base.encoding_manager.generate_random_organism(
            functional_length=self.ga_base.max_individual_length,
            include_specials=self.ga_base.delimiters,
            probability=0.10,
            verbose=False
        )
        self.visual_data_encoded.append(organism)
        self.visual_data_decoded.append(self.ga_base.decode_organism(organism))

        for cycle in range(1, self.mutation_cycles + 1):
            before = organism[:]
            organism, mutation_logs = self.ga_base.mutate_organism(
                organism, self.ga_base.current_generation, log_enhanced=True)
            self.log_enhanced_mutation(before, organism, mutation_logs, cycle)

            self.visual_data_encoded.append(organism)
            self.visual_data_decoded.append(self.ga_base.decode_organism(organism))

        self.print_logs()
        self.visualize()

# Configuration and instantiation
ga_base = M_E_GA_Base(
    genes=['A', 'C', 'G', 'T'],
    fitness_function=lambda x, _: -len(x),
    mutation_prob=0.053,
    delimited_mutation_prob=0.03,
    open_mutation_prob=0.003,
    capture_mutation_prob=0.05,
    delimiter_insert_prob=0.02,
    base_gene_prob=0.45,
    capture_gene_prob=0.05,
    max_individual_length=50,
    population_size=1,
    logging=True
)

experiment = Experiment(ga_base, seed=42, mutation_cycles=500)
experiment.run()
