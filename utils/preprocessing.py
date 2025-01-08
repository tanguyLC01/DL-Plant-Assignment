import cv2
import math
import numpy as np

def random_crop_opencv(image, target_size=(256, 256)):
    h, w, _ = image.shape
    scale = max(target_size[0] / h, target_size[1] / w)
    new_size = (int(w * scale), int(h * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    y_offset = np.random.randint(0, resized_image.shape[0] - target_size[0])
    x_offset = np.random.randint(0, resized_image.shape[1] - target_size[1])
    cropped_image = resized_image[y_offset:y_offset + target_size[0], x_offset:x_offset + target_size[1]]
    return cropped_image

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
        img = random_crop_opencv(img)
    img = enhance_img(img)
    return img