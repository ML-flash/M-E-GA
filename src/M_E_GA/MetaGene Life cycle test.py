from M_E_Engine import EncodingManager
from M_E_GA_Base import M_E_GA_Base
from MetaLifecycleFitness import MetaLifecycleFitness
import random


def run_lifecycle_experiment():
    # Define basic genes
    base_genes = ['A', 'B', 'C', 'D', 'E']

    # Initialize fitness function with length limit
    fitness_evaluator = MetaLifecycleFitness(
        target_pattern=['A', 'B', 'C', 'D', 'E'] * 30,
        max_length=80
    )

    # Configure GA
    ga = M_E_GA_Base(
        genes=base_genes,
        fitness_function=fitness_evaluator.calculate,  # Pass the method
        fitness_evaluator=fitness_evaluator,  # Store the instance
        mutation_prob=0.15,
        delimited_mutation_prob=0.10,
        open_mutation_prob=0.12,
        metagene_mutation_prob=0.03,
        delimiter_insert_prob=0.04,
        delimit_delete_prob=0.04,
        crossover_prob=0.50,
        elitism_ratio=0.10,
        base_gene_prob=0.30,
        metagene_prob=0.00,
        max_individual_length=50,
        population_size=500,
        num_parents=300,
        max_generations=1000,
        delimiters=False,
        delimiter_space=2,
        logging=False,
        experiment_name='meta_lifecycle_test',
        seed=None
    )

    # Set a small LRU cache size to force metagene turnover
    ga.encoding_manager.lru_cache_size = 50  # Reduced from 10 to make turnover more visible

    def monitor_metagenes(ga_instance):
        """Monitor and print metagene status each generation"""
        generation = ga_instance.current_generation
        num_metagenes = len(ga_instance.encoding_manager.meta_genes)
        in_basket = len(ga_instance.encoding_manager.deletion_basket)
        unused = len(ga_instance.encoding_manager.unused_encodings)

        print(f"\nGeneration {generation} Metagene Status:")
        print(f"Total Metagenes: {num_metagenes}")
        print(f"In Deletion Basket: {in_basket}")
        print(f"Unused Encodings: {unused}")

        if ga_instance.encoding_manager.meta_genes:
            print("\nExample Metagenes:")
            sample_size = min(3, len(ga_instance.encoding_manager.meta_genes))
            if sample_size > 0:
                for i, hash_key in enumerate(random.sample(
                        ga_instance.encoding_manager.meta_genes,
                        sample_size
                )):
                    decoded = ga_instance.encoding_manager.decode(
                        ga_instance.encoding_manager.encodings[hash_key]
                    )
                    print(f"Metagene {i + 1}: {decoded}")

        # Print best individual in current generation
        if hasattr(ga_instance, 'fitness_scores'):
            best_idx = ga_instance.fitness_scores.index(max(ga_instance.fitness_scores))
            best_individual = ga_instance.decode_organism(ga_instance.population[best_idx], format=True)
            print(f"\nBest Individual: {best_individual}")
            print(f"Best Fitness: {max(ga_instance.fitness_scores)}")

    ga.before_generation_finalize = monitor_metagenes

    # Run the experiment
    ga.run_algorithm()

    return ga


if __name__ == "__main__":
    print("Starting Meta Lifecycle Experiment...")
    ga_instance = run_lifecycle_experiment()

    print("\nExperiment Complete!")
    print(f"Final number of metagenes: {len(ga_instance.encoding_manager.meta_genes)}")
    print(f"Final unused encodings: {len(ga_instance.encoding_manager.unused_encodings)}")

    if hasattr(ga_instance, 'fitness_scores'):
        best_fitness = max(ga_instance.fitness_scores)
        avg_fitness = sum(ga_instance.fitness_scores) / len(ga_instance.fitness_scores)
        print(f"\nFinal Results:")
        print(f"Best Fitness: {best_fitness:.2f}")
        print(f"Average Fitness: {avg_fitness:.2f}")

        # Print final best solution
        best_idx = ga_instance.fitness_scores.index(best_fitness)
        best_individual = ga_instance.decode_organism(ga_instance.population[best_idx], format=True)
        print(f"\nBest Solution Found: {best_individual}")
        print(f"Target Pattern: {ga_instance.fitness_evaluator.target_pattern}")