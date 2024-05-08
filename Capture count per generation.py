import json
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.cm import get_cmap

def count_capture_mutations_per_generation(logs):
    capture_counts = []
    generations = []

    # Accessing each generation's log entry
    for generation_log in logs['logs']:
        generation_number = int(generation_log['generation'])
        generations.append(generation_number)

        # Ensure mutations list exists and skip 'None' entries safely
        mutations_list = generation_log.get('mutations', [])

        # Count 'capture' mutations in this generation, safely ignoring None entries
        capture_count = 0
        for mutation in mutations_list:
            if mutation and mutation.get('type') == 'capture':
                capture_count += 1
        capture_counts.append(capture_count)

    return generations, capture_counts

def aggregate_data_from_files(file_paths):
    # This will hold the capture counts and generations across all files for analysis
    all_generations = []
    all_capture_counts = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            logs = json.load(file)

        generations, capture_counts = count_capture_mutations_per_generation(logs)
        all_generations.extend(generations)
        all_capture_counts.extend(capture_counts)

    return all_generations, all_capture_counts

def plot_data(generations, capture_counts, num_runs):
    # Choose a colormap with enough colors for the maximum number of runs
    cmap = get_cmap('tab20')  # You can change 'tab20' to any other colormap you prefer

    # Set plot style to have black background
    plt.style.use('dark_background')

    # Plotting the aggregated data from multiple files
    for i in range(num_runs):
        start_idx = i * len(generations) // num_runs
        end_idx = (i + 1) * len(generations) // num_runs
        plt.scatter(generations[start_idx:end_idx], capture_counts[start_idx:end_idx],
                    color=cmap(i / num_runs), label=f'Run {i+1}', s=.3)  # Adjust the point size here

    plt.xlabel('Generation')
    plt.ylabel('Number of Capture Mutations')
    plt.title('Capture Mutations Across Generations from Multiple Runs')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Initialize Tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open file selection dialog for multiple files
    file_paths = filedialog.askopenfilenames(title="Select log files",
                                             filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
    if not file_paths:
        print("No files selected, exiting program.")
        return

    num_runs = len(file_paths)
    # Aggregate data for plotting
    generations, capture_counts = aggregate_data_from_files(file_paths)

    # Plotting the results
    plot_data(generations, capture_counts, num_runs)

if __name__ == "__main__":
    main()
