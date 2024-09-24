#TODO: functions which task will be to perform data preprocessing
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

scaler = MinMaxScaler(feature_range=(0,1))

# normalizes image size to 28x28
def normalize_image_size(img:np.ndarray) -> np.ndarray:
    return cv2.resize(img, (28,28))

# normalizes pixel values to a feature range of (0,1)
def normalize_pixel_values(img:np.ndarray) -> np.ndarray:
    normalized_img = scaler.fit_transform(img.reshape(-1, 1))
    return normalized_img.reshape(28,28,3)

# this function should be used to normalization
def normalize(img:np.ndarray) -> np.ndarray:
    return normalize_pixel_values(normalize_image_size(img))