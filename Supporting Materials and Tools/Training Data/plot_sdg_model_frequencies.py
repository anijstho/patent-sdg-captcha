import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Path to the pickle file
pickle_path = 'training_data.pkl'

# Load the pickle file
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

print(f"Data type: {type(data)}")
print(f"Shape: {data.shape}")

# Define models and SDGs
models = ['auckland', 'aurora', 'elsevier', 'sdgo', 'sdsn', 'siris', 'sdgbert', 'mistral']
sdgs = [f'sdg{i:02d}' for i in range(1, 17)]  # SDG01 to SDG16

# Initialize a DataFrame to store the frequencies
frequencies = pd.DataFrame(0, index=sdgs, columns=models)

# Count positives for each model and SDG
for sdg in sdgs:
    for model in models:
        column_name = f"{sdg}.{model}"
        if column_name in data.columns:
            if model == 'sdgbert':
                # For sdgbert, count values > 0.95 as positives
                frequencies.loc[sdg, model] = (data[column_name] > 0.95).sum()
            else:
                # For other models, count values == 1 as positives
                frequencies.loc[sdg, model] = (data[column_name] == 1).sum()

# Create the stacked bar graph
plt.figure(figsize=(10, 5))

# Create stacked bars for each SDG
bottom = np.zeros(len(sdgs))
for model in models:
    plt.bar(range(1, 17), frequencies[model], bottom=bottom, label=model, alpha=0.8)
    bottom += frequencies[model].values

# Add labels and modify axis settings
plt.ylabel('Number of Hits')
# No title as requested
plt.xticks(range(1, 17), [f'sdg{i:02d}' for i in range(1, 17)])  # SDG labels sdg01-sdg16
plt.legend(loc='upper left')  # Move legend to upper left

# Adjust y-axis to match the screenshot
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# Save the figure
plt.savefig('sdg_model_frequencies.png')

# Show the plot
plt.show()

print(f"Stacked bar graph created and saved as 'sdg_model_frequencies.png'")

# Also print the frequency table
print("\nFrequency Table:")
print(frequencies)