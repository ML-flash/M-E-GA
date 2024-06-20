class MegaGP:
    def __init__(self, input_size):
        self.input_size = input_size
        self.variables = {f"var{i}": None for i in range(input_size)}
        self.operator_priority = {'NOT': 3, 'AND': 2, 'OR': 1, 'XOR': 1}
        self.penalties = 0
        self.gene_penalties = 0  # Initialize gene penalties for nesting
        self.genes = 0
        self.successful_operations = 0  # Initialize successful operation counter

    def receive_input(self, binary_input):
        if len(binary_input) != self.input_size:
            raise ValueError("Input size mismatch.")
        for i, bit in enumerate(binary_input):
            self.variables[f"var{i}"] = bit

    def parse_organism(self, organism):
        output = []
        stack = []
        nesting_level = 0  # Track nesting level
        for token in organism:
            if token in self.variables:  # Variable
                output.append(token)
            elif token in self.operator_priority:  # Operator
                while (stack and stack[-1] != '(' and
                       self.operator_priority[stack[-1]] >= self.operator_priority[token]):
                    output.append(stack.pop())
                stack.append(token)
            elif token == '(':
                stack.append(token)
                nesting_level += 1
                self.genes += 1
                self.gene_penalties += nesting_level ** 1.01  # Increment gene penalties slightly exponentially
            elif token == ')':
                if stack and stack[-1] == '(':
                    stack.pop()  # Remove '('
                    self.penalties += 2  # Penalty for empty parentheses
                else:
                    while stack and stack[-1] != '(':
                        output.append(stack.pop())
                    if stack:
                        stack.pop()
                        nesting_level -= 1
                self.genes += 2.5
                self.gene_penalties += nesting_level ** 1.01  # Still apply nesting penalty
        # Empty the stack and check for unmatched '('
        while stack:
            top = stack.pop()
            if top == '(':
                self.penalties += 2  # Penalty for unmatched '('
            else:
                output.append(top)
        return output

    def evaluate_expression(self, expression):
        stack = []
        for token in expression:
            if token in self.variables:
                stack.append(self.variables[token])
            elif token in self.operator_priority:
                try:
                    if token == 'NOT':
                        operand = stack.pop()
                        stack.append(int(not operand))
                        self.successful_operations += 1
                    else:
                        right = stack.pop()
                        left = stack.pop()
                        result = self.calculate_operation(token, [left, right])
                        stack.append(result)
                        self.successful_operations += 1
                except IndexError:
                    self.penalties += 1
                    stack.append(0)
        return stack.pop() if stack else 0

    def calculate_operation(self, operator, operands):
        if operator == 'NOT':
            return int(not operands[0])
        elif operator == 'AND':
            return operands[0] and operands[1]
        elif operator == 'OR':
            return operands[0] or operands[1]
        elif operator == 'XOR':
            return operands[0] ^ operands[1]

    def evaluate_organism(self, organism, binary_input):
        self.receive_input(binary_input)
        parsed_expression = self.parse_organism(organism)
        output = self.evaluate_expression(parsed_expression)
        return output, self.penalties, self.successful_operations, self.gene_penalties

# Test Case
input_size = 8
mega = MegaGP(input_size)
organism = ['(', ')', 'var0', 'NOT', '(', 'var1', 'AND', 'var2', ')', '(', ')', 'XOR', '(', 'var3', ')']
binary_input = [1, 0, 1, 1, 0, 1, 0, 1]
output, penalties, successful_operations, gene_penalties = mega.evaluate_organism(organism, binary_input)

# Output Results
print("Output:", output)
print("Penalties:", penalties)
print("Successful Operations:", successful_operations)
print("Gene Penalties:", gene_penalties)

