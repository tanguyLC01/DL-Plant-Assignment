from utils.load_dataset import PlantVillageDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import cv2
from utils.preprocessing import preprocessing_img
import numpy as np
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3, InceptionResNetV2
import torchvision.models  as models

# Load the dataset
PATH = './Plant_leave_diseases_dataset_without_augmentation'
training_data = PlantVillageDataset(PATH, img_mode="LAB", train=True, transform=preprocessing_img)
test_data = PlantVillageDataset(PATH, img_mode="LAB", train=False, transform=preprocessing_img)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

resnet50v2 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
efficientnet_b0 = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
efficientnet_b3 = models.efficientnet_b3(weights='EfficientNet_B3_Weights.DEFAULT')
densenet = models.densenet121(weights='DenseNet121_Weights.DEFAULT')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet50v2 = resnet50v2.to(device)
efficientnet_b0 = efficientnet_b0.to(device)
efficientnet_b3 = efficientnet_b3.to(device)
densenet = densenet.to(device)

# Linear Model
import torch.nn as nn
class LinearHeadModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.25):
        super(LinearHeadModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
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
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.batch_norm(x)
        x = self.linear_layers(x)
        return x
    
def train_model(pre_trained_model, linear_model, train_loader, criterion, optimizer, epochs=50):
    linear_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            print(inputs.shape)
            optimizer.zero_grad()
            outputs_pre_trained = pre_trained_model(inputs)
            outputs = linear_model(outputs_pre_trained)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
        
resnet_linear_block = LinearHeadModel(1000, 38)
train_model(resnet50v2, resnet_linear_block, train_dataloader, nn.CrossEntropyLoss(), torch.optim.Adam(resnet50v2.parameters(), lr=0.001))