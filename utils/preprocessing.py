import cv2
import math
import numpy as np

def pad_to_square_opencv(image, target_size=(256, 256)):
    h, w, _ = image.shape
    desired_size = target_size[0]
    delta_w = desired_size - w
    delta_h = desired_size - h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return new_image

def enhance_img(img):
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    lab_clahe = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # AME Filter
    filtered = cv2.medianBlur(img_clahe, 3)
    
    # Gamma Correction
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    mid = 0.5
    mean = np.mean(val)
    gamma = math.log(mid*255)/math.log(mean)
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2RGB)
    
    return img_gamma2

def preprocessing_img(img):
    if img.shape != (256, 256, 3):
        img = pad_to_square_opencv(img)
    img = enhance_img(img)
    img = cv2.resize(img, (224, 224))
    img = enhance_img(img)
    return img