import cv2
import numpy as np

def rotation(img, angle):
    return cv2.rotate(img, angle)

def flip(img, flip_code):
    return cv2.flip(img, flip_code)

def gamma_correction(img, gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def noise_injection(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape)
    return np.clip(img + noise, 0, 255)

def scale(img, scale_factor):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor)  # Default interpolation is bilinear

def random_affine_transform(img):
    rows, cols = img.shape[:2]
    x_shift = np.random.randint(-50, 50)
    y_shift = np.random.randint(-50, 50)
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))  