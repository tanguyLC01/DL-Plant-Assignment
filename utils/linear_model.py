from utils.load_dataset import PlantVillageDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split, Subset
from utils.preprocessing import preprocessing_img
import torchvision.models  as models
from torchvision import transforms
import numpy as np
import cv2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Linear Model
import torch.nn as nn
class LinearHeadModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.25):
        super(LinearHeadModel, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.batch_norm(x)
        x = self.linear_layers(x)
        return x


steps = 100
def train_model(model, train_loader, test_loader, criterion, optimizer, device=device, epochs=10, steps=steps):
   
    for epoch in range(epochs):
        steps_count = steps
        model.train()
        running_loss = 0.0

        # Training Loop with mini-batches
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if steps_count == 0:
                break
            steps_count -= 1
            running_loss += loss.item()
        

        # Compute average training loss
        avg_train_loss = running_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        steps_count = steps
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if steps_count == 0:
                    break
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                steps_count -= 1

        test_accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def train_linear_model(index, 
                        input_dim, 
                        num_classes, 
                        models_list, 
                        train_models_dataloaders, 
                        test_models_dataloaders,
                        device=device):

    # Get pre-trained model
    pre_trained_model = models_list[index]

    # Set model to device
    pre_trained_model = pre_trained_model.to(device)

    # Create linear head model and add it to the pre-trained model
    model_linear_classifier = LinearHeadModel(input_dim, num_classes + 1)

    if index == 0:
        pre_trained_model.fc = model_linear_classifier.to(device)
        parameters = pre_trained_model.fc.parameters()
    else:
        pre_trained_model.classifier = model_linear_classifier.to(device)
        parameters = pre_trained_model.classifier.parameters()

    # Train and evaluate model
    train_model(
        pre_trained_model,
        train_models_dataloaders[index],
        test_models_dataloaders[index],
        nn.CrossEntropyLoss(),
        torch.optim.Adam(parameters, lr=0.001),
        device
    )

    if index == 0:
        return pre_trained_model.fc
    else:
        return pre_trained_model.classifier
