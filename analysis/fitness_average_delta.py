import json
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt


# Function to open a file dialog and select a file
def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


# Function to plot the graphs
def plot_fitness(log_data):
    generations = []
    average_fitness = []

    # Extracting the average fitness for each generation
    for generation_data in log_data:
        if 'summary' in generation_data and 'average_fitness' in generation_data['summary']:
            generations.append(generation_data['generation'])
            average_fitness.append(generation_data['summary']['average_fitness'])

    # Calculate fitness delta
    fitness_delta = [0] + [average_fitness[i] - average_fitness[i - 1] for i in range(1, len(average_fitness))]

    # Plotting the average fitness
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(generations, average_fitness, marker='o', linestyle='-', color='b')
    plt.title('Average Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')

    # Plotting the fitness delta
    plt.subplot(1, 2, 2)
    plt.plot(generations, fitness_delta, marker='o', linestyle='-', color='r')
    plt.title('Fitness Delta Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Delta')

    plt.tight_layout()
    plt.show()


# Main function
def main():
    file_path = open_file_dialog()
    if file_path:
        with open(file_path, 'r') as file:
            data = json.load(file)
            log_data = data['logs']
            plot_fitness(log_data)
    else:
        print("No file selected")


if __name__ == "__main__":
    main()