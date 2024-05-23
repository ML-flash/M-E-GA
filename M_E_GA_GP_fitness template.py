from mega_gp import MegaGP

class MegaGPFitnessFunction:
    def __init__(self, input_size):
        # Initialize the MegaGP with the specified input size
        self.mega_gp = MegaGP(input_size)
        # Extract genes directly from the MegaGP instance
        self.genes = self.extract_genes()

    def extract_genes(self):
        # Combine variable names and operators into a list of genes
        variables = list(self.mega_gp.variables.keys())
        operators = list(self.mega_gp.operator_priority.keys())
        parentheses = ['(', ')']  # Add parentheses as valid 'genes' for expression construction
        return variables + operators + parentheses

    def compute(self, organism, binary_input):
        # Evaluate the organism using the MegaGP instance
        output, penalties, successful_operations, gene_penalties = self.mega_gp.evaluate_organism(organism, binary_input)

        # Define a fitness score; here, you might prioritize operations and penalize for errors
        fitness_score = successful_operations - penalties - gene_penalties

        return fitness_score

# Example usage
input_size = 8
binary_input = [1, 0, 1, 1, 0, 1, 0, 1]  # Example input
organism = ['(', 'var0', 'NOT', 'var1', 'AND', 'var2', ')', 'XOR', 'var3']

# Initialize the fitness function class
fitness_function = MegaGPFitnessFunction(input_size)

# Compute the fitness for a given organism and binary input
fitness_score = fitness_function.compute(organism, binary_input)
print("Fitness Score:", fitness_score)
