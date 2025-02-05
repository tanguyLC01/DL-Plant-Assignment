# DL - Plant Assignment

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

## Transfer Learning
We loaded each of the selected models:
- DenseNet201 (Huang et al., 2017),
- EfficientNetB0 (Tan and Le, 2019),
- InceptionResNetV2 (Szegedy et al., 2017),
- EfficientNetB3 (Tan and Le, 2019)

Then we modified the last layer of the models and replaced it with a simple linear model, before training it on the dataset. We used batches and early stopping for computing limitation reasons.

## 1. Preprocessing the images
In the first approach, we preprocessed the images using:
- CLAHE
- AME
- Gamma Correction
- Square standard size for all images

## 2. Data augmentation
In the second approach, we augmented the dataset using:
-  flipping,
-  rotation,
-  gamma correction,
-  noise injection,
-  scaling,
-  affine transform

## 3. Transformer
Lastly, we used the ViT-based-patch16-224-in21k transformer from Google and fine-tuned the ViT.

# Results
