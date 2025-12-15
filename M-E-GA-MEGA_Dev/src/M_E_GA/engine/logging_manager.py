"""
logging_manager.py

Manages the local generation-level logs for M_E_GA_Base and integrates with GA_Logger
for real-time event-based logging.

Part of the refactoring for Single Responsibility Principle (SRP).
"""


class LoggingManager:
    """
    LoggingManager handles the creation and storage of structured logs during the
    genetic algorithm runs. It works in tandem with a GA_Logger (if provided)
    for real-time event-based logging.

    Attributes:
        logging_enabled (bool): Master switch for logging. If False, no logs are generated.
        generation_logging (bool): Whether to log generation summaries.
        mutation_logging (bool): Whether to log mutation events.
        crossover_logging (bool): Whether to log crossover events.
        individual_logging (bool): Whether to log individual fitness each generation.
        logger (GA_Logger or None): Optional event-logger instance for real-time event logs.
        logs (list): A list where each element is a dict representing a single generation’s log.
    """

    def __init__(self,
                 logging_enabled,
                 generation_logging,
                 mutation_logging,
                 crossover_logging,
                 individual_logging,
                 logger=None):
        """
        Initialize the LoggingManager.

        :param logging_enabled: Master boolean controlling whether logs should be recorded at all.
        :param generation_logging: If True, log generation summaries.
        :param mutation_logging: If True, log detailed mutation info.
        :param crossover_logging: If True, log crossover events.
        :param individual_logging: If True, log each individual's fitness data per generation.
        :param logger: Optional GA_Logger instance for real-time event logging.
        """
        self.logging_enabled = logging_enabled
        self.generation_logging = generation_logging
        self.mutation_logging = mutation_logging
        self.crossover_logging = crossover_logging
        self.individual_logging = individual_logging
        self.logger = logger

        # Local generation-level logs
        self.logs = []

    def start_new_generation_logging(self, generation_number):
        """
        Begin logging info for a new generation by appending a new "generation" record.

        :param generation_number: The current generation index.
        """
        if not self.logging_enabled:
            return

        generation_log = {
            "generation": generation_number,
            "summary": {},
            "individuals": [],
            "mutations": [],
            "crossovers": [],
            "organisms": []
        }
        self.logs.append(generation_log)

    def log_generation(self, generation, fitness_scores, population=None):
        """
        Record summary statistics for a generation, if enabled.

        :param generation: The generation index.
        :param fitness_scores: List of fitness scores for the generation.
        :param population: The population of encoded organisms (optional).
        """
        if not self.logging_enabled or not self.generation_logging:
            return

        average_fitness = sum(fitness_scores) / len(fitness_scores)
        sorted_scores = sorted(fitness_scores)
        median_fitness = sorted_scores[len(sorted_scores) // 2]
        best_fitness = max(fitness_scores)
        worst_fitness = min(fitness_scores)

        summary_log = {
            "average_fitness": average_fitness,
            "median_fitness": median_fitness,
            "best_fitness": best_fitness,
            "worst_fitness": worst_fitness,
        }

        # Log event-based if we have a GA_Logger
        if self.logger:
            self.logger.log_event("generation_summary", summary_log)

        current_generation_log = self.logs[-1]
        current_generation_log["summary"] = summary_log

    def log_mutation(self, mutation_details):
        """
        Record a mutation event, if mutation logging is enabled.

        :param mutation_details: A dict describing the mutation (type, index, generation, etc.).
        """
        if not self.logging_enabled or not self.mutation_logging:
            return

        if self.logger:
            self.logger.log_event("mutation", mutation_details)

        if self.logs:
            current_generation_log = self.logs[-1]
            current_generation_log["mutations"].append(mutation_details)

    def log_crossover(self, generation, parent1, parent2, crossover_point, offspring1, offspring2):
        """
        Record a crossover event, if crossover logging is enabled.

        :param generation: The current generation index.
        :param parent1: The first parent's organism encoding.
        :param parent2: The second parent's organism encoding.
        :param crossover_point: The index at which crossover occurred (can be None).
        :param offspring1: The first offspring's organism encoding.
        :param offspring2: The second offspring's organism encoding.
        """
        if not self.logging_enabled or not self.crossover_logging:
            return

        crossover_log = {
            "generation": generation,
            "crossover_point": crossover_point,
            "parent1": parent1,
            "parent2": parent2,
            "offspring1": offspring1,
            "offspring2": offspring2,
        }

        if self.logger:
            self.logger.log_event("crossover", crossover_log)

        if self.logs:
            current_generation_log = self.logs[-1]
            current_generation_log["crossovers"].append(crossover_log)

    def log_new_organism(self, organism_encoding):
        """
        Log a newly created organism to the current generation log, if logs exist.

        :param organism_encoding: The encoded representation of the new organism.
        """
        if not self.logging_enabled or not self.logs:
            return

        self.logs[-1]["organisms"].append({"encoding": organism_encoding})

    def log_organism_state(self, stage, organism, generation):
        """
        Log the state of an organism at a given stage (e.g., 'before_mutation').

        :param stage: A string describing the stage, e.g. 'before_mutation'.
        :param organism: The encoded organism list.
        :param generation: The current generation index.
        """
        if not self.logging_enabled or not self.logs:
            return

        organism_log = {
            "stage": stage,
            "generation": generation,
            "encoded_organism": organism.copy()
        }
        self.logs[-1]["organisms"].append(organism_log)

    def individual_logging_fitness(self, generation, population, fitness_scores):
        """
        Log per-individual fitness data if enabled.

        :param generation: The current generation index.
        :param population: The list of encoded organisms for this generation.
        :param fitness_scores: The list of fitness scores for the population.
        """
        if not self.logging_enabled or not self.individual_logging:
            return

        current_generation_log = self.logs[-1]
        for idx, fitness_score in enumerate(fitness_scores):
            current_generation_log["individuals"].append({
                "individual_index": idx,
                "organism": population[idx],
                "fitness_score": fitness_score
            })

    def get_logs(self):
        """
        Get the entire list of generation logs.

        :return: A list of logs, each representing one generation.
        """
        return self.logs
