from mega_gp import MegaGP
from itertools import product
import math

class MegaGPFitnessFunction:
    def __init__(self, input_size, mega_gp):
        # Initialize the MegaGP with the specified input size
        self.mega_gp = mega_gp

        # Extract genes directly from the MegaGP instance
        self.genes = self.extract_genes()

        # Assuming input_size refers to the total number of bits (selection lines + data lines)
        # Calculate the number of selection lines for the MUX that would fit these criteria
        self.selection_lines = math.floor(math.log2(input_size))
        self.truth_table = self.generate_mux_truth_table(self.selection_lines)

    def extract_genes(self):
        # Combine variable names and operators into a list of genes
        variables = list(self.mega_gp.variables.keys())
        operators = list(self.mega_gp.operator_priority.keys())
        parentheses = ['(', ')']  # Add parentheses as valid 'genes' for expression construction
        return variables + operators + parentheses

    def generate_mux_truth_table(self, selection_lines):
        # Number of input signals
        num_inputs = 2 ** selection_lines
        
        # Generate all possible input combinations
        inputs = list(product([0, 1], repeat=num_inputs))
        
        # Generate all possible selection values
        selections = list(product([0, 1], repeat=selection_lines))
        
        # Initialize the truth table
        truth_table = {}
        
        # Evaluate each combination of input signals and selection values
        for selection_value in selections:
            # Calculate the output index from the selection value
            index = int(''.join(str(x) for x in selection_value), 2)
            # Store the input-output pair
            truth_table[tuple(selection_value)] = [inp[index] for inp in inputs]
        
        return truth_table

    def compute(self, encoded_individual):
        # Decode the individual into a format suitable for evaluation
        decoded_individual = self.decode_organism(encoded_individual)

        # Initialize the score for correct outputs
        total_correct_outputs = 0
        total_tests = len(self.truth_table)

        # Test the organism against each input in the truth table
        for selection_value, expected_outputs in self.truth_table.items():
            # Use the decoded individual to get the output from the MegaGP interpreter
            output, penalties, successful_operations, gene_penalties = self.mega_gp.evaluate_organism(decoded_individual, selection_value)

            # Reassign penalties and successful operations for now
            # Note: These values are being calculated but not used in the current fitness score calculation

            # Check the output against the expected output from the truth table
            if output == expected_outputs:
                total_correct_outputs += 1

        # Calculate the fitness score as the percentage of correct outputs
        fitness_score = total_correct_outputs / total_tests
        return fitness_score

    def decode_organism(self, encoded_individual):
        # Assuming the GA instance has a method to decode organisms
        return ga_instance.decode_organism(encoded_individual)
