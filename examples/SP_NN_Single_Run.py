import random
import numpy as np
from M_E_GA import M_E_GA_Base, M_E_Engine
from SP_NN_Fitness import NetworkEvolutionFitness


class ExperimentRunner:
    def __init__(self, debug: bool = False, config_file=None):
        self.debug = debug

        # Network configuration - core structural parameters only
        # Input/Output neuron counts are managed by fitness class based on simulation requirements
        self.network_params = {
            # Core network structure
            'volume_size': 10.0,
            'total_neurons': 70,

            # Neuron radius parameters
            'max_radius': 2.0,
            'min_radius': 1.50,
            'hidden_radius_range': (1.0, 0.50),
            'base_radius_shrink_rate': 0.95,
            'input_radius_factor': .60,
            'interface_radius_factor': .60,
            'interface_offset': 1.0,

            # Activation parameters
            'activation_budget': 1000,
            'time_window_size': 50,
            'activation_threshold': 0.5,
            'activation_radius_factor': 0.2
        }

        # GA configuration with updated probability parameter names
        self.ga_config = {
            'mutation_prob': 0.15,
            'delimited_mutation_prob': 0.10,
            'open_metagene_mutation_prob': 0.05,        # Updated from 'open_mutation_prob'
            'capture_metagene_mutation_prob': 0.03,     # Updated from 'capture_mutation_prob'
            'insert_delimiter_pair_prob': 0.04,         # Updated from 'delimiter_insert_prob'
            'delete_delimiter_prob': 0.05,              # Updated from 'delimit_delete_prob'
            'crossover_prob': 0.00,
            'elitism_ratio': 0.00,
            'base_gene_prob': 0.30,
            'capture_metagene_prob': 0.03,              # Updated from 'capture_gene_prob'
            'max_individual_length': 100,
            'population_size': 500,
            'num_parents': 200,
            'max_generations': 1000,
            'delimiters': False,
            'delimiter_space': 2,
            'logging': False,
            'experiment_name': 'neural_evolution',
            'seed': None
        }

        # Best solution tracking
        self.best_organism = {
            "genome": None,
            "fitness": float('-inf')
        }

    def update_best_organism(self, genome, fitness, verbose=True):
        if fitness > self.best_organism["fitness"]:
            self.best_organism["genome"] = genome
            self.best_organism["fitness"] = fitness
            if verbose:
                print(f"New best fitness: {fitness}")
                self.print_network_stats()

    def print_network_stats(self):
        """Print detailed network statistics for debugging"""
        if hasattr(self, 'fitness_function'):
            stats = self.fitness_function.get_stats()
            if self.debug:
                print("\nNetwork Statistics:")
                print(f"Connectivity: {stats['connectivity']['current']:.2f}%")
                print(f"Unreachable neurons: {stats['connectivity']['unreachable_neurons']}")
                print("\nNeuron Distribution:")
                print(f"Total neurons: {stats['network']['total_neurons']}")
                print(f"Input neurons: {stats['network']['input_neurons']}")
                print(f"Hidden neurons: {stats['network']['hidden_neurons']}")
                print(f"Output neurons: {stats['network']['output_neurons']}")
                print("\nActivation Parameters:")
                print(f"Activation budget: {self.network_params['activation_budget']}")
                print(f"Time window: {self.network_params['time_window_size']}")

    def setup_experiment(self):
        # Create fitness function with update callback
        self.fitness_function = NetworkEvolutionFitness(
            network_params=self.network_params,
            update_best_func=self.update_best_organism,
            max_path_length=70,
            path_step_reward=1.0,
            debug=self.debug  # Pass debug flag to fitness function
        )

        # Initialize GA with updated ga_config
        self.ga = M_E_GA_Base(
            genes=self.fitness_function.genes,
            fitness_function=lambda ind, ga_instance: self.fitness_function.compute(ind, ga_instance),
            **self.ga_config
        )

    def run_experiment(self):
        if self.debug:
            print("Starting Neural Evolution Experiment...")
            print("\nCore Network Parameters:")
            for key, value in self.network_params.items():
                print(f"  {key}: {value}")
            print("\nNote: Input/Output neuron counts are managed by fitness class")
            print(f"\nPopulation size: {self.ga_config['population_size']}")
            print(f"Max generations: {self.ga_config['max_generations']}\n")

        # Run the GA
        self.ga.run_algorithm()

        # Get results
        best_genome = self.best_organism["genome"]
        best_fitness = self.best_organism["fitness"]
        best_solution = self.ga.decode_organism(best_genome, format=True) if best_genome is not None else []

        # Only print final results regardless of debug setting
        print("\nExperiment Results:")
        print(f"Best Solution: {best_solution}")
        print(f"Best Fitness: {best_fitness}")
        print(f"Solution Length: {len(best_solution)}")

        if self.debug:
            self.print_network_stats()  # Print final network statistics only in debug mode

        return {
            'best_genome': best_genome,
            'best_fitness': best_fitness,
            'best_solution': best_solution
        }


if __name__ == "__main__":
    # Create and run experiment with debug flag
    debug_mode = False  # Set to True to enable debug output
    experiment = ExperimentRunner(debug=debug_mode)
    experiment.setup_experiment()
    results = experiment.run_experiment()
