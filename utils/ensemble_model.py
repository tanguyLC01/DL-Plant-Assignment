from utils.load_dataset import PlantVillageDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from utils.preprocessing import preprocessing_img
import torchvision.models  as models
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class WeightedEnsemble(nn.Module):
    def __init__(self, models, num_classes):
        super(WeightedEnsemble, self).__init__()
        self.models = models
        self.num_models = len(models) 
        self.weights = nn.Parameter(torch.ones(self.num_models, device=device) / self.num_models)
        self.num_classes = num_classes
        for model in models:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        # Move inputs to the MPS device
        inputs = inputs.to('mps')  

        # Initialize all_preds as a tensor on the MPS device
        all_preds = torch.empty((len(self.models), inputs.size(0), self.num_classes), device='mps')

        # Forward pass for each model in the ensemble
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)  # Outputs from the model

                # Apply softmax to get probabilities
                prob = nn.functional.softmax(outputs, dim=1)

                # Store predictions in the pre-allocated tensor
                all_preds[i] = prob

        # Normalize the weights (softmax ensures weights sum to 1)
        normalized_weights = nn.functional.softmax(self.weights, dim=0).to('mps')

        # Aggregate the weighted predictions
        weighted_preds = torch.sum(all_preds * normalized_weights.view(-1, 1, 1), dim=0)
        return weighted_preds



def train_ensemble_weights(ensemble, criterion, optimizer, train_dataloader, test_dataloader):

    ensemble.train()  # Ensemble training mode (weights can be trained)
    running_loss = 0.0

    steps_count = 1000
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Zero gradients for optimizer
        outputs = ensemble(inputs).to(device)  # Forward pass through ensemble
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights
        
        running_loss += loss.item()

        if steps_count == 0:
            break
        steps_count -= 1
        
    # Evaluate on test set
    ensemble.eval()
    correct = 0
    total = 0
    steps_count = 1000
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = ensemble(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if steps_count == 0:
                break
            steps_count -= 1

    test_accuracy = correct / total

    print(f"Loss: {running_loss/len(train_dataloader):.4f}, Test Accuracy: {test_accuracy:.4f}")
