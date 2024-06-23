import random

class MegaGP:
    def __init__(self, input_size):
        self.input_size = input_size
        self.variables = {f"var{i}": None for i in range(input_size)}
        self.operators = ['NOT', 'AND', 'OR', 'XOR']
        self.penalties = 0
        self.gene_penalties = 0
        self.genes = 0
        self.successful_operations = 0

    def receive_input(self, binary_input):
        if len(binary_input) != self.input_size:
            raise ValueError("Input size mismatch.")
        for i, bit in enumerate(binary_input):
            self.variables[f"var{i}"] = bit

    def parse_organism(self, organism):
        self.genes = len(organism)
        return organism

    def evaluate_expression(self, expression):
        stack = []
        for token in expression:
            if token in self.variables:
                stack.append(self.variables[token])
            elif token == 'NOT':
                try:
                    operand = stack.pop()
                    stack.append(int(not operand))
                    self.successful_operations += 1
                except IndexError:
                    self.penalties += 1
                    stack.append(0)
            elif token in ['AND', 'OR', 'XOR']:
                try:
                    right = stack.pop()
                    left = stack.pop()
                    if token == 'AND':
                        result = left and right
                    elif token == 'OR':
                        result = left or right
                    elif token == 'XOR':
                        result = left ^ right
                    stack.append(result)
                    self.successful_operations += 1
                except IndexError:
                    self.penalties += 1
                    stack.append(0)
            else:
                self.penalties += 1
                stack.append(0)
        return stack.pop() if stack else 0

    def evaluate_organism(self, organism, binary_input):
        self.penalties = 0
        self.gene_penalties = 0
        self.genes = 0
        self.successful_operations = 0
        self.receive_input(binary_input)
        parsed_expression = self.parse_organism(organism)
        output = self.evaluate_expression(parsed_expression)
        return output, self.penalties, self.successful_operations, self.genes

    def generate_random_organism(self, length):
        variables = list(self.variables.keys())
        organism = random.choices(variables + self.operators, k=length)
        return organism

# Test Case
input_size = 8
organism_length = 15
mega = MegaGP(input_size)

# Generate a random organism
random_organism = mega.generate_random_organism(organism_length)
binary_input = [random.randint(0, 1) for _ in range(input_size)]

# Evaluate the random organism
output, penalties, successful_operations, genes = mega.evaluate_organism(random_organism, binary_input)

# Output Results
print("Random Organism:", ['var2', 'var3', 'OR', 'AND', 'var0', 'NOT', 'var0', 'var0', 'OR', 'var2', 'OR', 'var0', 'var2', 'var1', 'NOT', 'var1', 'var1', 'var5', 'var1', 'var1', 'var2', 'NOT', 'var0', 'XOR', 'var1', 'var0', 'OR', 'var2'])
print("Binary Input:", binary_input)
print("Output:", output)
print("Penalties:", penalties)
print("Successful Operations:", successful_operations)
print("Genes:", genes)