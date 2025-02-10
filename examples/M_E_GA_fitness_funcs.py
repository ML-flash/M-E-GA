class LeadingOnesFitness:
    def __init__(self, max_length, update_best_func):
        self.max_length = max_length  # Maximum length of an individual for scoring normalization
        self.update_best = update_best_func  # Function to update the best organism
        self.genes = ['0', '1']  # Define the specific genes used in this fitness function

    def compute(self, encoded_individual, ga_instance):
        """
        Computes the fitness of an individual based on the number of leading ones without penalties.

        Args:
            encoded_individual (str or list of str): The encoded individual to evaluate. If a string, it's treated as a single individual; if a list, it's assumed to be multiple individuals.
            ga_instance (object): An instance of the genetic algorithm framework. This is used for decoding and other operations depending on the context.

        Returns:
            float or list of floats: The computed fitness scores for each encoded individual. If a single individual is passed, returns a float; if multiple individuals are passed, returns a list of floats.
        """
        decoded_individual = ga_instance.decode_organism(encoded_individual)  # Decode the individual
        fitness_score = 0

        # Count the number of leading '1's until the first '0' or the end of the sequence
        for gene in decoded_individual:
            if gene == '1':
                fitness_score += 1
            else:
                break  # Stop counting at the first '0'

        # Normalize the fitness score to be between 0 and self.max_length
        normalized_fitness = min(fitness_score, self.max_length)

        # Update the best organism using the passed function
        self.update_best(encoded_individual, normalized_fitness, verbose=True)

        # Return the final fitness score
        return normalized_fitness
