# DL - Plant Assignment -

## Project Description
This project was inspired by the paper _An ensemble of deep learning architectures for accurate plant
disease classification_ by Ali Hussein Ali et al., Basra University.
The goal was to classify different plant diseases based on images. We used Transfer Learning from four pre-trained models: DenseNet201, EfficientNetB0, InceptionResNetV2 and EfficientNetB3. Then we combined their outputs using Ensemble Learning.
In this repertory, we studied three different approaches and reported the results we obtained:
  1. Preprocessing the images with CLAHE, AME and Gamma correction
  2. Augmenting the dataset
  3. Using a Transformer

## Dataset

You can find dataset of unaugmented leaf images at this address : https://data.mendeley.com/datasets/tywbtsjrjv/1. It is composed of 39 different classes of plant leaf and it contains 61,486 images.

## Pipeline of the project
- Load the PlantVillage Dataset
- Enhance images with CLAHE, AME and Gamma correction
- Transfer Learning with DenseNet201 (Huang et al., 2017), EfficientNetB0 (Tan and Le, 2019),
InceptionResNetV2 (Szegedy et al., 2017), EfficientNetB3 (Tan and Le,
2019), and ResNet50v2
- Ensembe Learning to benefit from the 5 feature extraction models
- Output : 38 Classes belonging-probabilities
