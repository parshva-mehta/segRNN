import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to the TensorBoard event file
event_file = r"logs\segrnn_experiment\version_8\events.out.tfevents.1733182487.nid001133.1864088.1"

# Output CSV file path
output_csv = "tensorboard_scalars_version_8.csv"

# Load the event file
event_acc = EventAccumulator(event_file)
event_acc.Reload()

# Get all scalar tags
tags = event_acc.Tags()["scalars"]

# Open a CSV file for writing
with open(output_csv, "w", newline="") as csv_file:
   writer = csv.writer(csv_file)
   writer.writerow(["Tag", "Step", "Value"])  # Header row

   # Iterate over tags and write each scalar to the CSV
   for tag in tags:
      scalar_events = event_acc.Scalars(tag)
      for event in scalar_events:
         writer.writerow([tag, event.step, event.value])

print(f"Scalars exported to {output_csv}")
