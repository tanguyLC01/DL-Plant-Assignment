from utils.load_dataset import PlantVillageDataset
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader

# Load the dataset
PATH = './Plant_leave_diseases_dataset_without_augmentation'
training_data = PlantVillageDataset(PATH, "LAB", train=True)
test_data = PlantVillageDataset(PATH, "LAB", train=False)

# Let's plot some image
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(list(training_data.label_to_idx.keys())[label])
    plt.axis("off")
    plt.imshow(img, cmap="gray")
plt.show()

# Data Loaders
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


os.system('Pause')