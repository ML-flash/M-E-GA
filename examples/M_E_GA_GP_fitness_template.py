from M_E_GA_GP_interpreter import MegaGP
from itertools import product
import math


class MegaGPFitnessFunction:
    def __init__(self, approximate_input_size, update_best_func):
        self.selection_lines = self.find_nearest_valid_selection_lines(approximate_input_size)
        self.data_lines = 2 ** self.selection_lines
        total_input_size = self.selection_lines + self.data_lines

        print(f"Selected Configuration: {self.selection_lines} selection lines, {self.data_lines} data lines")
        print(f"Total Input Size: {total_input_size} (as required by MegaGP)")

        self.mega_gp = MegaGP(total_input_size)
        self.update_best = update_best_func
        self.genes = self.extract_genes()
        self.truth_table = self.generate_mux_truth_table(self.selection_lines)

    def find_nearest_valid_selection_lines(self, approx_size):
        closest_size = float('inf')
        best_selection = 0
        for S in range(1, int(math.log2(approx_size)) + 1):
            total_size = S + (2 ** S)
            if abs(total_size - approx_size) < abs(closest_size - approx_size):
                closest_size = total_size
                best_selection = S
            if total_size > approx_size:  # Stop if the size surpasses the approximate size
                break
        return best_selection

    def extract_genes(self):
        variables = list(self.mega_gp.variables)
        operators = list(self.mega_gp.operators)
        return variables + operators

    def generate_mux_truth_table(self, selection_lines):
        num_inputs = 2 ** selection_lines
        inputs = list(product([0, 1], repeat=num_inputs))
        selections = list(product([0, 1], repeat=selection_lines))
        truth_table = []

        for inp in inputs:
            for selection_value in selections:
                index = int(''.join(str(x) for x in selection_value), 2)
                output = inp[index]
                truth_table.append((inp, selection_value, output))

        return truth_table

    def compute(self, encoded_individual, ga_instance, verbose=False):
        decoded_individual = ga_instance.decode_organism(encoded_individual, format=True)
        total_correct_outputs = 0
        total_correct = 0
        total_penalties = 0
        total_tests = 0

        for inp, selection_value, expected_output in self.truth_table:
            total_tests +=1
            # Generate binary input based on selection lines and data lines
            binary_input = list(selection_value) + list(inp)

            # Evaluate the organism
            output, penalties, successful_operations, gene_penalties = self.mega_gp.evaluate_organism(
                decoded_individual, binary_input)


            # Calculate net penalties
            net_penalties = penalties + (gene_penalties/1.4) - successful_operations

            # Check correctness of the output
            if output == expected_output:
                total_correct_outputs += 10
                total_correct += 1

            total_penalties += net_penalties

            # Verbose output
            if verbose:
                print(f"Testing input: {binary_input}")
                print(f"Expected output: {expected_output}, Actual output: {output}")
                print(f"Penalties: {penalties}, Gene Penalties: {gene_penalties}, Successful Operations: {successful_operations}")
                print(f"Net Penalties: {net_penalties}")

        # Calculate the average penalties per test
        avg_penalties = total_penalties / total_tests

        # Fitness score calculation
        correct_output_score = total_correct_outputs
        penalty_score = avg_penalties


        fitness_score = correct_output_score - penalty_score

        if verbose:
            print(f"Total Correct Outputs: {total_correct}/{total_tests}")
            print(f"Total Penalties: {total_penalties}")
            print(f"Average Penalties: {avg_penalties}")
            print(f"Correct Output Score: {correct_output_score}, Penalty Score: {penalty_score}")
            print(f"Final Fitness Score: {fitness_score}")

        self.update_best(encoded_individual, fitness_score, verbose=False)
        return fitness_score

# Example Usage
#fitness_function = MegaGPFitnessFunction(approximate_input_size=8)
