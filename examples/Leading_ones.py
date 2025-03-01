# Leading_ones.py

class LeadingOnesFitness:
    def __init__(self, max_length, update_best_func):
        self.max_length = max_length
        self.update_best = update_best_func  # Store the function for updating the best organism
        self.genes = ['0', '1']  # Allowed genes
        self.target_gene = '1'  # Start by optimizing for leading 1s
        self.phase = 1  # Tracks which phase we're in (1: maximizing '1's, 2: maximizing '0's)

    def compute(self, encoded_individual, ga_instance):
        # Decode the individual
        decoded_individual = ga_instance.decode_organism(encoded_individual)

        # Initialize fitness score
        fitness_score = 0

        # Count leading target genes (either '1' or '0' depending on phase)
        for i, gene in enumerate(decoded_individual):
            if i >= self.max_length:
                break  # Stop counting at max_length

            if gene == self.target_gene:
                fitness_score += 1
            else:
                break  # Stop at first mismatch

        # If we reached the target max_length, switch optimization goal
        if fitness_score == self.max_length:
            self.phase = 2 if self.phase == 1 else 1  # Toggle between phases
            self.target_gene = '0' if self.phase == 2 else '1'  # Update target

        # Update the best organism using the passed function
        self.update_best(encoded_individual, fitness_score, verbose=True)

        # Return the final fitness score
        return fitness_score
