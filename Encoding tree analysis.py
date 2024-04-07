import tkinter as tk
from tkinter import filedialog
import json
from collections import defaultdict


def load_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        log_data = json.load(file)
    return log_data


def reconstruct_gene_tree(encodings):
    base_genes = {v: k for k, v in encodings.items() if isinstance(v, str) and v not in ['Start', 'End']}
    captured_segments = {k: v for k, v in encodings.items() if isinstance(v, list)}

    gene_tree = defaultdict(list)

    for k, v in base_genes.items():
        gene_tree[k].append(v)

    for k, segment in captured_segments.items():
        for item in segment:
            for gene, hash_key in base_genes.items():
                if item == hash_key:
                    gene_tree[gene].append(k)
                    break
            else:
                for parent_key, parent_segment in captured_segments.items():
                    if item in parent_segment:
                        gene_tree[parent_key].append(k)
                        break

    return gene_tree


def print_gene_tree_iteratively(gene_tree, max_depth=3):
    stack = [(node, 0) for node in gene_tree.keys()]
    stack.reverse()

    visited = set()

    while stack:
        node, depth = stack.pop()

        if depth > max_depth:
            continue

        if node in visited:
            continue
        visited.add(node)

        indent = "    " * depth
        print(f"{indent}{node}")

        if node in gene_tree:
            for child in reversed(gene_tree[node]):
                stack.append((child, depth + 1))


# Initialize Tkinter root
root = tk.Tk()
root.withdraw()  # Hide the main window

# Open file selection dialog
log_file_path = filedialog.askopenfilename(title="Select a log file",
                                           filetypes=(("JSON files", "*.json"), ("All files", "*.*")))

# Proceed if a file was selected
if log_file_path:
    log_data = load_log_file(log_file_path)  # Load the selected log file

    final_encodings = log_data.get('final_encodings', {})  # Safely get final encodings

    gene_tree = reconstruct_gene_tree(final_encodings)  # Reconstruct the gene tree from encodings

    print_gene_tree_iteratively(gene_tree)  # Print a subset of the gene tree
