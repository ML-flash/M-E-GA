# M_E_GA_fitness_funcs.py
import random

class LeadingOnesFitness:
    def __init__(self, max_length, update_best_func):
        """
        LeadingOnesFitness is just here for completeness in the same file.
        """
        self.max_length = max_length
        self.update_best_func = update_best_func
        self.genes = ["0", "1"]

    def compute(self, individual, ga_instance):
        decoded = ga_instance.decode_organism(individual, format=True)
        score = 0
        for gene in decoded:
            if gene == "1":
                score += 1
            else:
                break
        score = score
        self.update_best_func(individual, score, verbose=True)
        return score

import math


class NQueensFitness:
    def __init__(self, n, update_best_func, base_threshold=None):
        """
        Initialize the fitness function for the N-Queens problem.

        :param n: The board size (each board must have n rows).
        :param update_best_func: A callback function with signature:
            update_best_func(individual, fitness, complete_count, row_count, verbose=True)
        :param base_threshold: The base number of rows that receive full reward.
                              Defaults to 2*n if not specified.
        """
        self.n = n
        self.update_best_func = update_best_func
        self.base_threshold = base_threshold if base_threshold is not None else 2 * n
        self.genes = list(range(1, n + 1))
        # We cache only complete boards that have no conflicts.
        self._board_cache = {}  # Key: tuple(board); Value: conflict count (which will be 0)

    def count_conflicts(self, board):
        """
        Count the total number of pairwise conflicts in a board.
        Only cache the result if the board is complete and conflict-free.

        :param board: A list of integers representing queen positions in the board.
        :return: The total number of conflicting pairs.
        """
        board_tuple = tuple(board)
        m = len(board)
        # Only use the cache for complete boards.
        if m == self.n and board_tuple in self._board_cache:
            return self._board_cache[board_tuple]

        # Compute conflicts normally.
        if m < 2:
            return 0

        col_counts = {}
        diag1_counts = {}  # key = row - col
        diag2_counts = {}  # key = row + col

        for i, col in enumerate(board):
            col_counts[col] = col_counts.get(col, 0) + 1
            d1 = i - col
            diag1_counts[d1] = diag1_counts.get(d1, 0) + 1
            d2 = i + col
            diag2_counts[d2] = diag2_counts.get(d2, 0) + 1

        conflicts = 0
        for dct in (col_counts, diag1_counts, diag2_counts):
            for k in dct.values():
                if k > 1:
                    conflicts += (k * (k - 1)) // 2

        # Only cache if the board is complete and conflict-free.
        if m == self.n and conflicts == 0:
            self._board_cache[board_tuple] = conflicts

        return conflicts

    def compute(self, individual, ga_instance):
        """
        Compute the fitness of an individual.

        The chromosome is interpreted as a sequence of rows.
        Every consecutive block of n numbers is a board state.

        Global Row Reward:
          - Each row gives +1 if its index is less than T.
          - Each row past T gives -2.
          T = base_threshold + (complete_count * n)

        Board Scoring:
          - For a complete board (length n):
              * If conflict-free and unique (i.e. not seen before in this individual), add bonus n.
              * Otherwise, score = n - (# conflicts).
          - For an incomplete board (length < n):
              Score = m - (# conflicts) - (n - m).

        Total fitness = (global row reward) + (sum of board scores).

        The update callback is called with (individual, total_fitness, complete_count, row_count, verbose=True).
        """
        n = self.n
        L = len(individual)  # Total number of rows in the chromosome

        # Partition individual into boards.
        boards = []
        for i in range(0, L, n):
            boards.append(individual[i:i + n])

        board_score = 0
        complete_count = 0
        unique_complete_boards = []

        for board in boards:
            m = len(board)
            missing = (n - m) if m < n else 0
            conflicts = self.count_conflicts(board)
            if m == n:
                if conflicts == 0:
                    bonus = n  # Bonus reward for a complete, conflict-free board.
                    board_tuple = tuple(board)
                    if board_tuple not in unique_complete_boards:
                        unique_complete_boards.append(board_tuple)
                    else:
                        bonus = 0  # Duplicate complete board: no bonus.
                    complete_count += 1
                    board_contribution = bonus - conflicts
                else:
                    board_contribution = n - conflicts
            else:
                board_contribution = m - conflicts - missing
            board_score += board_contribution

        # Dynamic threshold T: base_threshold + (complete_count * n)
        T = self.base_threshold + complete_count * n

        # Global row reward: each row i (0-indexed) gets:
        #   +1 if i < T, else -2.
        global_row_reward = 0
        for i in range(L):
            if i < T:
                global_row_reward += 1
            else:
                global_row_reward -= 2

        total_fitness = global_row_reward + board_score

        self.update_best_func(individual, total_fitness, complete_count, L, verbose=True)
        return total_fitness


