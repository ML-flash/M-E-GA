import random
from M_E_GA_Base_V2 import M_E_GA_Base


GLOBAL_SEED =            None
num_cycles =              1
MAX_GENERATIONS =        500
random.seed(GLOBAL_SEED)


MUTATION_PROB =           0.06
DELIMITED_MUTATION_PROB = 0.04
OPEN_MUTATION_PROB =      0.007
CAPTURE_MUTATION_PROB =   0.001
DELIMITER_INSERT_PROB =   0.004
CROSSOVER_PROB =          .90
ELITISM_RATIO =           0.5
BASE_GENE_PROB =          0.30
MAX_INDIVIDUAL_LENGTH =   100
POPULATION_SIZE =         600
NUM_PARENTS =             100
DELIMITER_SPACE =         3
DELIMITERS =              False


LOGGING =                 False
GENERATION_LOGGING =      True
MUTATION_LOGGING =        False
CROSSOVER_LOGGING =       False
INDIVIDUAL_LOGGING =      True


NUM_TASKS = 45
NUM_MACHINES = 15
NUM_JOBS = 15
TASKS_PER_JOB = 3
AVAILABLE_TIME = 3000

#SPECIAL maCHINE VALUES
MIN_USAGE_LIMIT =          2
MAX_USAGE_LIMIT =          10
SPECIAL_PERCENTAGE =       0.10

HARD_LIMIT =   100  #Hard limit for the organism size.
BASE_PENALTY = 1  # Base for the exponential penalty, adjust as needed
MIN_PENALTY =  0  #Lowest the penalty goes

best_organism = {
    "genome": None,
    "fitness": float('-inf')  # Start with negative infinity to ensure any valid organism will surpass it
}


def calculate_penalty(encoded_length, decoded_length):
    if encoded_length == 0:  # Prevent division by zero
        return BASE_PENALTY

    efficiency_ratio = decoded_length / encoded_length
    penalty = BASE_PENALTY

    if decoded_length <= HARD_LIMIT:
        # As efficiency improves, reduce the penalty, but not below the min_penalty
        if efficiency_ratio > 1:
            penalty /= efficiency_ratio  # Decrease penalty for higher efficiency
            penalty = max(penalty, MIN_PENALTY)  # Enforce a minimum penalty
    else:
        # For decoded lengths exceeding the hard limit, increase penalty smoothly
        excess_ratio = decoded_length / HARD_LIMIT
        penalty *= excess_ratio * 1.4 # Increase penalty based on how much the length exceeds the limit

    return penalty

def update_best_organism(current_genome, current_fitness, verbose ):
    verbose = True
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")

# Define tasks with machine-specific processing times
def create_tasks(num_tasks, machine_names):
    tasks_with_processing_times = {
        f"Task_{task_index}": {machine_name: random.randint(15, 40) for machine_name in machine_names}
        for task_index in range(num_tasks)
    }
    return tasks_with_processing_times

def create_jobs(num_jobs, tasks_per_job, available_tasks):
    job_details = {}
    all_task_keys = list(available_tasks.keys())
    total_required_tasks = num_jobs * tasks_per_job

    if total_required_tasks > len(all_task_keys):
        # Not enough tasks to go around, some jobs will be marked as non-existent
        tasks_for_jobs = random.sample(all_task_keys, len(all_task_keys))  # Use all available tasks
        remaining_tasks = len(all_task_keys)
    else:
        tasks_for_jobs = random.sample(all_task_keys, total_required_tasks)  # Sample the exact number needed
        remaining_tasks = total_required_tasks

    for job_index in range(num_jobs):
        if remaining_tasks >= tasks_per_job:
            # Assign a unique set of tasks to this job
            assigned_tasks = tasks_for_jobs[:tasks_per_job]
            tasks_for_jobs = tasks_for_jobs[tasks_per_job:]
            remaining_tasks -= tasks_per_job
        else:
            # Mark this job as having a non-existent set of tasks
            assigned_tasks = []

        job_details[f"Job_{job_index}"] = {
            'tasks': assigned_tasks,
            'value': random.randint(10, 200) if assigned_tasks else 0  # No value for jobs with non-existent tasks
        }

    return job_details


def create_machines():
    efficent_machines = {'Machine_Very_Efficient': {
        'usage_limit': 10,
        'efficiency_multiplier': 0.5
    }, 'Machine_Less_Efficient': {
        'usage_limit': 20,
        'efficiency_multiplier': 2
    }}
    # Start with two machines with predefined properties
    num_special_machines = max(2, int(NUM_MACHINES * SPECIAL_PERCENTAGE))  # Ensure at least two special machines, including the two predefined ones

    # Create additional special machines if needed
    for i in range(2, num_special_machines):
        machine_identifier = f"Machine_Special_{i}"
        usage_limit = random.randint(MIN_USAGE_LIMIT, MAX_USAGE_LIMIT // 2)  # Lower half of the range for special machines
        efficiency_multiplier = random.uniform(0.1, 0.5)  # Higher efficiency for special machines
        efficent_machines[machine_identifier] = {
            'usage_limit': usage_limit,
            'efficiency_multiplier': efficiency_multiplier
        }

    # Create regular machines
    for machine_index in range(num_special_machines, NUM_MACHINES):
        usage_limit = random.randint(MIN_USAGE_LIMIT, MAX_USAGE_LIMIT)
        # Regular efficiency multiplier calculation
        efficiency_multiplier = 2 - (usage_limit - MIN_USAGE_LIMIT) / (MAX_USAGE_LIMIT - MIN_USAGE_LIMIT) * (1.9 - 0.1)
        machine_identifier = f"Machine_{machine_index}"
        efficent_machines[machine_identifier] = {
            'usage_limit': usage_limit,
            'efficiency_multiplier': efficiency_multiplier
        }

    return efficent_machines

def allocate_task(machine, task, state, machines):
    if state['machine_usages'].get(machine, 0) < machines[machine]['usage_limit']:
        state['machine_usages'][machine] = state['machine_usages'].get(machine, 0) + 1
        # Use calculate_task_time to get adjusted task time
        adjusted_time = calculate_task_time(machine, state['tasks'][task], machines)
        # Update the state with the adjusted time and other necessary changes
        # For example, update the total time spent, mark task as completed, etc.
        # state['total_time'] += adjusted_time
        # state['completed_tasks'].add(task)
        return adjusted_time  # Return the adjusted time for further processing
    else:
        # Handle case where machine's usage limit is reached by returning an infeasible time
        return 10000000  # Return an infeasible time to indicate this allocation is not possible

def calculate_task_time(machine, task_time, machines):
    efficiency_multiplier = machines[machine]['efficiency_multiplier']
    adjusted_time = task_time * efficiency_multiplier
    return adjusted_time


# Initialize global variables for tasks, jobs, and machines
machines = create_machines()
tasks = create_tasks(NUM_TASKS, machines)
jobs = create_jobs(NUM_JOBS, TASKS_PER_JOB, tasks)
GENES = list(tasks.keys()) + list(machines.keys())

# Problem-specific fitness function
def problem_specific_fitness_function(encoded_genome, ga_instance, jobs, tasks, machines, available_time, verbose=False):
    global global_best_individual, global_best_fitness
    decoded_genome = ga_instance.decode_organism(encoded_genome)
    encoded_length = len(encoded_genome)
    decoded_length = len(decoded_genome)
    time_elapsed, fitness_score = 0, 0
    completed_tasks = set()
    machine_usages = {machine: 0 for machine in machines}  # Track machine usage
    expecting_machine = False
    current_task = None
    skipped_task = 0
    current_job_contributions = {job_id: 0 for job_id in jobs}  # Track contributions to job completion

    for gene in decoded_genome:
        # Adjust lengths for 'Start' and 'End' markers
        if gene in ['Start', 'End']:
            decoded_length -= 1
            encoded_length -= 1
            continue

        if expecting_machine:
            if gene in machines and current_task and gene in tasks[current_task] and machine_usages[gene] < \
                    machines[gene]['usage_limit']:
                task_time = calculate_task_time(gene, tasks[current_task][gene], machines)
                if time_elapsed + task_time <= available_time:
                    time_elapsed += task_time
                    machine_usages[gene] += 1
                    completed_tasks.add(current_task)

                    # Base reward for completing a task
                    task_reward = 0.01

                    # Check if the task belongs to any job and apply additional reward if it contributes to job completion
                    for job_id, job_details in jobs.items():
                        if current_task in job_details['tasks']:
                            # The task contributes to a job, so it's worth double
                            task_reward *= 2

                            # Update job contributions
                            current_job_contributions[job_id] += 1 / len(job_details['tasks'])
                            if current_job_contributions[job_id] == 1:  # Job fully completed
                                fitness_score += job_details['value']  # Full job value reward
                                current_job_contributions[job_id] = 0  # Reset job contribution
                                if verbose:
                                    print(f"Completed job {job_id}, added value: {job_details['value']}")

                    # Add the task reward to the fitness score
                    fitness_score += task_reward

                    if verbose:
                        print(
                            f"Completed task {current_task} using {gene}, task time: {task_time}, total time elapsed: {time_elapsed}")

                    expecting_machine = False
            else:
                if verbose and current_task:
                    print(f"Expected machine for {current_task}, but found {gene}. Skipping task.")
                    skipped_task += 1
                expecting_machine = False  # Reset if no valid machine follows a task
        elif gene in tasks:
            current_task = gene
            expecting_machine = True
            if verbose:
                print(f"Task queued: {current_task}, awaiting machine allocation...")

    if time_elapsed > available_time:
        fitness_score = max(0, fitness_score)  # Ensure fitness is not negative
        if verbose:
            print(f"Exceeded available time, total time elapsed: {time_elapsed}")

    # Apply penalties based on genome lengths
    penalty = (.003 * decoded_length)
    penalty += skipped_task * 3

    fitness_score -= penalty
    #fitness_score
    if verbose:
        print(f"Applied penalty: {penalty}, current fitness score: {fitness_score}")

    update_best_organism(encoded_genome, fitness_score, verbose)

    return fitness_score, {
        "tasks": tasks,
        "machines": machines,
        "jobs": jobs,
        "time_elapsed": time_elapsed,
        "completed_tasks": completed_tasks
    }


# GA configuration
config = {
    'genes': GENES,
    'fitness_function': lambda ind, ga: problem_specific_fitness_function(ind, ga, jobs, tasks, machines, available_time = AVAILABLE_TIME),
    'mutation_prob': MUTATION_PROB,
    'delimited_mutation_prob': DELIMITED_MUTATION_PROB,
    'open_mutation_prob': OPEN_MUTATION_PROB,
    'capture_mutation_prob': CAPTURE_MUTATION_PROB,
    'delimiter_insert_prob': DELIMITER_INSERT_PROB,
    'crossover_prob': CROSSOVER_PROB,
    'elitism_ratio': ELITISM_RATIO,
    'base_gene_prob': BASE_GENE_PROB,
    'max_individual_length': MAX_INDIVIDUAL_LENGTH,
    'population_size': POPULATION_SIZE,
    'num_parents': NUM_PARENTS,
    'max_generations': MAX_GENERATIONS,
    'delimiters': DELIMITERS,
    'delimiter_space': DELIMITER_SPACE,
    'logging': LOGGING,
    'generation_logging': GENERATION_LOGGING,
    'mutation_logging': MUTATION_LOGGING,
    'crossover_logging': CROSSOVER_LOGGING,
    'individual_logging': INDIVIDUAL_LOGGING,
    'seed': GLOBAL_SEED
}

# Initialize GA
ga = M_E_GA_Base(**config)

# Execute the GA
ga.run_algorithm()

# Analyze the results
best_genome = best_organism["genome"]
best_fitness = best_organism["fitness"]
best_solution_decoded = ga.decode_organism(best_genome, format=True)
print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
verbose_fitness_score, _ = problem_specific_fitness_function(best_genome, ga, jobs, tasks, machines, AVAILABLE_TIME, verbose=True)
print(f"Verbose Fitness Evaluation Score: {verbose_fitness_score}")
print(f"Best Genome (Encoded): {best_genome}")
