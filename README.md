# DL - Plant Assignment -

## Download Dataset

You can find dataset of unaugmented leaf images at this address : https://data.mendeley.com/datasets/tywbtsjrjv/1

## Pipeline of the project
- Load the PlantVillage Dataset
- Enhance images with CLAHE, AME and Gamma correction
- Transfer Learning with DenseNet201 (Huang et al., 2017), EfficientNetB0 (Tan and Le, 2019),
InceptionResNetV2 (Szegedy et al., 2017), EfficientNetB3 (Tan and Le,
2019), and ResNet50v2
- Ensembe Learning to benefit from the 5 feature extraction models
- Output : 38 Classes belonging-probabilities
