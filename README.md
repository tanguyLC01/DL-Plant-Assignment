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

Then we modified the last layer of the models and replaced it with a simple linear model :
4 Dense Layer (256 -> 128 -> 64 -> 1), all activated by ReLU and with a Dropout Rate by 0.25 between the 3 first layers.
Training Process : Batch Size = 64, Early stop at 100 steps and epochs = 10.

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

In this approach, we did not use the preprocessing techniques.

## 3. Transformer
Lastly, we used the ViT-based-patch16-224-in21k transformer from Google and fine-tuned the ViT with 6 epochs.

# Results & Discussion
| Technics        | Train Loss       | Test Accuracy  |
| --------------- |:----------------:| --------------:|
| Preprocessed    |     0.5282       |    0.8981      |
| Augmented       |     0.0148       |    0.8706      |
| Transformer     |     0.000286     |    0.9872      |

# How to Use
If you want to see the models specifications, there are all in the folder _utils_ in the files _linear_model.py_, _ensemble_learning_.
The preprocessing techniques : _utils > preprocessing.py_
To run the data augmentation, run in console : python run_augmentatio.py at the root level of the project
The models training are in _model_augmented.ipynb, model_non_augmented.ipynb, ViT > create_XVit.ipynb_
  
