from utils.load_dataset import PlantVillageDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split, Subset
from utils.preprocessing import preprocessing_img
import torchvision.models  as models
from torchvision import transforms
import numpy as np


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the dataset
PATH = './Plant_leave_diseases_dataset_without_augmentation'
transform = transforms.Compose([
    preprocessing_img,
    transforms.ToTensor(),
])
training_data = PlantVillageDataset(PATH, img_mode="LAB", train=True, transform=transform, test_split=0.3)
validation_data  = PlantVillageDataset(PATH, img_mode="LAB", train=False, transform=transform, test_split=0.3)


# Create sets and dataloaders for the 4 models
n_models = 4
indices = np.random.permutation(len(training_data))
model_sets = [Subset(training_data, indices[i::n_models]) for i in range(n_models)]
test_size = int(0.2 * len(model_sets[0]))  # 20% for testing
train_size = len(model_sets[0]) - test_size  # Remaining 80%
train_set = []
test_set = []
for i in range(n_models):
    train, test = random_split(model_sets[i], [train_size, test_size])
    train_set.append(train)
    test_set.append(test)

train_models_dataloaders = [DataLoader(train_data, batch_size=64, shuffle=True) for train_data in train_set ]
test_models_dataloaders = [DataLoader(test_data, batch_size=64, shuffle=True) for test_data in test_set ]


# create sets and loaders for the ensemble learning
test_size = int(0.2 * len(validation_data))
train_validation_set, test_validation_set = random_split(validation_data, [len(validation_data) - test_size, test_size])

train_validation_dataloader = DataLoader(train_validation_set, batch_size=64, shuffle=True)
test_validation_dataloader = DataLoader(test_validation_set, batch_size=64, shuffle=True)