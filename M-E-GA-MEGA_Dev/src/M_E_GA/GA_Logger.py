import datetime
import json
import os


class GA_Logger:
    def __init__(self, experiment_name, log_directory="MEGA Logs"):
        """
        Initialize the GA_Logger instance.

        :param experiment_name: The name of the experiment, used for generating log filenames.
        :param log_directory: The directory where log files will be stored. Defaults to "MEGA Logs" in the user's home directory.
        """
        self.experiment_name = experiment_name
        self.log_directory = os.path.join(os.path.expanduser("~"), log_directory)
        self.events = []  # List to store all logged events.
        self.subscribers = []  # List of subscriber callback functions for real-time event notifications.

    def subscribe(self, callback):
        """
        Subscribe a callback function to receive real-time event notifications.

        :param callback: A function that takes a single argument (event dictionary) and processes it.
        """
        self.subscribers.append(callback)

    def log_event(self, event_type, details):
        """
        Record an event and notify all subscribers.

        :param event_type: A string indicating the type of event (e.g., "generation_summary", "mutation").
        :param details: A dictionary containing additional details about the event.
        """
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self.events.append(event)

        # Notify all subscriber callbacks of the new event.
        for callback in self.subscribers:
            try:
                callback(event)
            except Exception as e:
                print("Error in subscriber callback:", e)

    def save(self, filename=None):
        """
        Save all collected events to a JSON file.

        :param filename: Optional filename. If not provided, a filename is generated using the experiment name and current timestamp.
        """
        if filename is None:
            filename = f"{self.experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

        # Ensure the log directory exists.
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        full_path = os.path.join(self.log_directory, filename)
        with open(full_path, 'w') as f:
            json.dump(self.events, f, indent=4)
        print(f"Logs saved to {full_path}")
