import utils.augmentation as aug
from utils.load_dataset import PlantVillageDataset
import os
import cv2
import numpy as np

path = './Plant_leave_diseases_dataset_without_augmentation'
new_path = './Plant_leave_diseases_dataset_with_augmentation'

def save_image(parent_path, idx, img):
    cv2.imwrite(os.path.join(parent_path, f'image ({idx}).JPG'), img)
    
os.makedirs(new_path, exist_ok=True)
for idx, label in enumerate(sorted(os.listdir(path))):
    old_label_dir = os.path.join(path, label)
    new_label_dir = os.path.join(new_path, label)
    os.makedirs(new_label_dir, exist_ok=True)
    i = 0
    print(f'Augmenting {label}...')
    for img_name in os.listdir(old_label_dir):
        img_path = os.path.join(old_label_dir, img_name)
        image = cv2.imread(img_path)
        for angle in np.arange(-15, 15, 5):
            save_image(new_label_dir, i, aug.rotation(image, angle))
            i += 1
        for flip_code in [0, 1, -1]:
            save_image(new_label_dir, i, aug.flip(image, flip_code))
            i += 1
        for gamma in np.arange(0.7, 1.3, 0.1):
            save_image(new_label_dir, i, aug.gamma_correction(image, gamma))
            i += 1
        for _ in range(6):
            save_image(new_label_dir, i, aug.noise_injection(image,mean=0, std=1e-2))
            i += 1
        for scale_factor in np.arange(0.8, 1.2, 0.05):
            save_image(new_label_dir, i, aug.scale(image, scale_factor))
            i += 1
        for _ in range(8):
            save_image(new_label_dir, i, aug.random_affine_transform(image))
            i += 1
    print(f'Augmented {label} with {i} images')


 