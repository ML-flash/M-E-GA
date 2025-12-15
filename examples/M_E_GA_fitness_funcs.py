# M_E_GA_fitness_funcs.py

class LeadingOnesFitness:
    def __init__(self, max_length, update_best_func):
        self.max_length = max_length
        self.update_best = update_best_func  # Store the passed function for updating the best organism
        self.genes = ['0', '1']  # Specific genes for this fitness function

    def compute(self, encoded_individual, ga_instance):
        # Decode the individual
        decoded_individual = ga_instance.decode_organism(encoded_individual)

        # Initialize fitness score
        fitness_score = 0

        # Count the number of leading '1's until the first '0'
        for gene in decoded_individual:
            if gene == '1':
                fitness_score += 1
            else:
                break  # Stop counting at the first '0'

        # Calculate the penalty
        if len(decoded_individual) < self.max_length:
            penalty = 1.008 ** (self.max_length - len(decoded_individual))
        else:
            penalty = len(decoded_individual) - self.max_length

        # Compute final fitness score
        final_fitness = fitness_score - penalty

        # Update the best organism using the passed function
        self.update_best(encoded_individual, final_fitness, verbose=True)

        # Return the final fitness score after applying the penalty
        return final_fitness
