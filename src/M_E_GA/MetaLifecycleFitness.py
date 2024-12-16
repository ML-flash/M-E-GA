class MetaLifecycleFitness:
    def __init__(self, target_pattern=['A', 'B', 'C', 'D', 'E'] * 4, max_length=50):
        self.target_pattern = target_pattern
        self.pattern_length = len(target_pattern)
        self.max_length = max_length

    def calculate(self, encoded_individual, ga_instance):
        """Calculate fitness based on longest matching sequence from start"""
        decoded_sequence = ga_instance.decode_organism(encoded_individual, format=True)

        # Hard penalty for exceeding max length
        if len(decoded_sequence) > self.max_length:
            return 1.0  # Minimal fitness for too-long sequences

        # Count matching elements from start
        matching_length = 0
        for i in range(min(len(decoded_sequence), len(self.target_pattern))):
            if decoded_sequence[i] == self.target_pattern[i]:
                matching_length += 1
            else:
                break

        # Linear reward instead of exponential to prevent overflow
        fitness = matching_length * 10.0

        return fitness

    def evaluate(self, population, ga_instance):
        """Evaluate entire population"""
        return [self.calculate(ind, ga_instance) for ind in population]