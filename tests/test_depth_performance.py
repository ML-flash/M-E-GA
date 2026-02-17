"""
Tests for the _compute_depths optimization: correctness and performance.
"""

import random
import time
import unittest

from src.M_E_GA import M_E_GA_Base
from src.M_E_GA.engine.mutation_manager import MutationManager


class TestComputeDepthsCorrectness(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        self.ga = M_E_GA_Base(
            genes=['A', 'B', 'C'],
            fitness_function=lambda org, ga: 0.0,
            population_size=2,
            max_individual_length=10,
            logging=False,
            experiment_name="TestDepths"
        )
        self.mm = self.ga.mutation_manager
        self.start = self.ga.encoding_manager.reverse_encodings['Start']
        self.end = self.ga.encoding_manager.reverse_encodings['End']

    def test_matches_original_no_delimiters(self):
        a = self.ga.encoding_manager.reverse_encodings['A']
        b = self.ga.encoding_manager.reverse_encodings['B']
        organism = [a, b, a, b, a]
        depths = MutationManager._compute_depths(organism, self.start, self.end)
        for i in range(len(organism)):
            self.assertEqual(depths[i], self.mm.calculate_depth(organism, i))

    def test_matches_original_with_delimiters(self):
        a = self.ga.encoding_manager.reverse_encodings['A']
        organism = [self.start, a, a, self.end, a, self.start, a, self.end]
        depths = MutationManager._compute_depths(organism, self.start, self.end)
        for i in range(len(organism)):
            self.assertEqual(depths[i], self.mm.calculate_depth(organism, i))

    def test_matches_original_nested(self):
        a = self.ga.encoding_manager.reverse_encodings['A']
        organism = [self.start, a, self.start, a, self.end, a, self.end]
        depths = MutationManager._compute_depths(organism, self.start, self.end)
        for i in range(len(organism)):
            self.assertEqual(depths[i], self.mm.calculate_depth(organism, i))

    def test_matches_original_unmatched(self):
        a = self.ga.encoding_manager.reverse_encodings['A']
        organism = [self.end, a, self.start, a, self.end, self.end, a]
        depths = MutationManager._compute_depths(organism, self.start, self.end)
        for i in range(len(organism)):
            self.assertEqual(depths[i], self.mm.calculate_depth(organism, i))

    def test_matches_original_random_organisms(self):
        population = self.ga.initialize_population()
        for organism in population:
            depths = MutationManager._compute_depths(organism, self.start, self.end)
            for i in range(len(organism)):
                self.assertEqual(depths[i], self.mm.calculate_depth(organism, i))

    def test_empty_organism(self):
        self.assertEqual(MutationManager._compute_depths([], self.start, self.end), [])


class TestComputeDepthsPerformance(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        self.ga = M_E_GA_Base(
            genes=['A', 'B', 'C'],
            fitness_function=lambda org, ga: 0.0,
            population_size=2,
            max_individual_length=10,
            logging=False,
            experiment_name="TestDepthsPerf"
        )
        self.mm = self.ga.mutation_manager
        self.start = self.ga.encoding_manager.reverse_encodings['Start']
        self.end = self.ga.encoding_manager.reverse_encodings['End']

    def test_precompute_is_faster(self):
        a = self.ga.encoding_manager.reverse_encodings['A']
        codons = [a, self.start, a, self.end]
        organism = codons * 250  # length 1000

        # Old approach: call calculate_depth(organism, i) for every i
        t0 = time.perf_counter()
        for _ in range(10):
            for i in range(len(organism)):
                self.mm.calculate_depth(organism, i)
        old_time = time.perf_counter() - t0

        # New approach: single _compute_depths call
        t0 = time.perf_counter()
        for _ in range(10):
            MutationManager._compute_depths(organism, self.start, self.end)
        new_time = time.perf_counter() - t0

        speedup = old_time / new_time
        print(f"\nOld (per-index): {old_time:.4f}s | New (precompute): {new_time:.4f}s | Speedup: {speedup:.1f}x")
        self.assertGreater(speedup, 10, f"Expected >10x speedup, got {speedup:.1f}x")


if __name__ == '__main__':
    unittest.main()
