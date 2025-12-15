import matplotlib.pyplot as plt
import networkx as nx
import threading
import queue
from collections import defaultdict


class MetagenomePlotter:
    def __init__(self):
        """Initialize the real-time metagenome plotter."""
        self.plot_queue = queue.Queue()
        self.running = True

        # Initialize matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.setup_plot()

        # Threading lock
        self._lock = threading.Lock()

        # Cache for metagene orders
        self.order_cache = {}

    def setup_plot(self):
        """Set up the initial plot configuration."""
        self.ax.set_title('Metagenome Evolution')
        self.ax.set_xlabel('Metagene Order')
        self.ax.set_ylabel('Count')
        self.ax.grid(True)
        plt.tight_layout()

    def compute_metagene_order(self, mg, encoding_manager):
        """
        Compute the order of a metagene (recursive).
        Base genes are order 0, metagenes containing only base genes are order 1, etc.
        """
        # Check cache first
        if mg in self.order_cache:
            return self.order_cache[mg]

        encoding = encoding_manager.encodings.get(mg, ())
        orders = []

        for element in encoding:
            if element in encoding_manager.meta_genes:
                child_order = self.compute_metagene_order(element, encoding_manager)
                orders.append(child_order)
            else:
                orders.append(0)  # Base genes are order 0

        order = max(orders, default=0) + 1
        self.order_cache[mg] = order
        return order

    def analyze_metagenome(self, encoding_manager):
        """
        Analyze the current state of the metagenome.
        Returns counts of metagenes at each order level.
        """
        order_counts = defaultdict(int)

        # Reset cache for new analysis
        self.order_cache = {}

        # Count base genes (order 0)
        base_genes = set(encoding_manager.reverse_encodings.keys()) - set(['Start', 'End'])
        base_genes = base_genes - set(encoding_manager.meta_genes)
        order_counts[0] = len(base_genes)

        # Analyze metagenes
        for mg in encoding_manager.meta_genes:
            order = self.compute_metagene_order(mg, encoding_manager)
            order_counts[order] += 1

        return dict(order_counts)

    def update_plot(self, order_counts):
        """Update the plot with new metagene order counts."""
        self.ax.clear()

        # Plot data
        orders = sorted(order_counts.keys())
        counts = [order_counts[order] for order in orders]

        # Create bar plot
        bars = self.ax.bar(orders, counts, color='skyblue', alpha=0.7)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}',
                         ha='center', va='bottom')

        # Customize plot
        self.ax.set_title('Metagenome Evolution')
        self.ax.set_xlabel('Metagene Order')
        self.ax.set_ylabel('Count')
        self.ax.grid(True, alpha=0.3)

        # Set x-axis to show only integer values
        self.ax.set_xticks(orders)

        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def handle_event(self, event):
        """Handle incoming events from the GA_Logger."""
        if event['event_type'] == 'generation_summary':
            # Queue an update request
            self.plot_queue.put("update")

    def _update_loop(self):
        """Main loop for updating the plot."""
        while self.running:
            try:
                # Wait for update signal with timeout
                msg = self.plot_queue.get(timeout=0.1)
                if msg == "update":
                    with self._lock:
                        # Get current state from encoding manager
                        order_counts = self.analyze_metagenome(self.ga_instance.encoding_manager)
                        self.update_plot(order_counts)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in update loop: {e}")

    def start(self, ga_instance):
        """
        Start the plotter and connect it to the GA instance.

        Args:
            ga_instance: The GA instance to monitor
        """
        self.ga_instance = ga_instance

        if ga_instance.logger:
            ga_instance.logger.subscribe(self.handle_event)

        # Start the update loop in a separate thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def stop(self):
        """Stop the plotter and clean up."""
        self.running = False
        plt.close(self.fig)


# Example usage:
"""
# Initialize your GA instance as before
ga = M_E_GA_Base(
    genes=['A', 'B', 'C'],
    fitness_function=your_fitness_function,
    experiment_name="Test Experiment",
    logging=True,
    generation_logging=True
)

# Create and start the plotter
plotter = MetagenomePlotter()
plotter.start(ga)

# Run your GA
ga.run_algorithm()

# When done
plotter.stop()
"""