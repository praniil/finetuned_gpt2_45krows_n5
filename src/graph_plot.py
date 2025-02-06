import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

log_dir = '/home/nil/python_projects/gpt2_finetuned_45k_10epochs/logs' 

base_save_dir = '/home/nil/python_projects/gpt2_finetuned_45k_10epochs/graphs'

# Load event files
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# List all available scalar tags
scalar_tags = ea.Tags().get('scalars', [])
print(f"Found scalar tags: {scalar_tags}")

# Loop through each scalar tag and plot/save the graph
for tag in scalar_tags:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label=tag, linewidth=2)  

    # Add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add minor ticks for finer grid control
    plt.minorticks_on()

    # Labels and Title
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel(tag.split('/')[-1].capitalize(), fontsize=12)
    plt.title(f'{tag.split("/")[-1].capitalize()} over Training Steps', fontsize=14)
    plt.legend()

    tag_path = tag.replace('/', '_') 
    save_path = os.path.join(base_save_dir, f'{tag_path}_plot.png')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the plot
    plt.savefig(save_path)
    plt.close()

    print(f"Graph for '{tag}' saved at: {save_path}")
