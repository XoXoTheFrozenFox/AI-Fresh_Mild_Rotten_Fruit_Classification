import os
import matplotlib.pyplot as plt

# Path to the dataset folder
dataset_path = r'C:/CODE/Code/CODE ON GITHUB/AI-Fresh_Mild_Rotten_Fruit_Classification/Dataset3/train'

# Dictionary to store class names and their respective image counts
class_counts = {}

# Iterate through each folder in the dataset path
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):  # Check if it's a directory
        # Count the number of files in the directory
        image_count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        class_counts[class_name] = image_count

# Prepare data for plotting
classes = list(class_counts.keys())
counts = list(class_counts.values())

# Plot the histogram
plt.figure(figsize=(12, 8))
bars = plt.bar(classes, counts, color='skyblue')

# Add text labels on each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x position of the text
        yval + 0.5,  # y position of the text (slightly above the bar)
        int(yval),  # Text value
        ha='center',  # Horizontal alignment
        va='bottom',  # Vertical alignment
        fontsize=10
    )

plt.xlabel('Class Name')
plt.ylabel('Number of Photos')
plt.title('Number of Photos per Class for Train set')
plt.xticks(rotation=45, ha='right')  # Rotate class names for better readability
plt.tight_layout()
plt.show()