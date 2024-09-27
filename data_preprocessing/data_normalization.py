import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

scaler = MinMaxScaler(feature_range=(0,1))


def normalize_image_size(img:np.ndarray) -> np.ndarray:
    """Resizes any given image to 28x28 (size of mnist dataset)

    Args:
        img (np.ndarray): original image bitmap

    Returns:
        np.ndarray: resized image bitmap
    """
    return cv2.resize(img, (28,28))


def normalize_pixel_values(img:np.ndarray) -> np.ndarray:
    """Normalizes pixel values in the given image with a feature range (0,1)

    Args:
        img (np.ndarray): original image bitmap

    Returns:
        np.ndarray: pixel-normalized image bitmap
    """
    normalized_img = scaler.fit_transform(img.reshape(-1, 1))
    return normalized_img.reshape(28,28)


def normalize(img:np.ndarray) -> np.ndarray:
    """Normalizes an image to match mnist dataset (color_code, pixel_values, size)

    Args:
        img (np.ndarray): original image bitmap

    Returns:
        np.ndarray: normalized image bitmap
    """
    
    #TODO: Bitwise 'not' operation must be applied (Mnist dataset is on a black page, predicted numbers will be mostly on white page)
    #TODO: Conversion to grayscale format needs to be done
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = normalize_image_size(img)
    img = normalize_pixel_values(img)

    return img

def normalize_mnist_dataset(dataset:np.ndarray) -> np.ndarray:
    normalized_dataset= []

    for i in tqdm(range(len(dataset)), desc='Loading Mnist...', colour='#00FF00'):
        normalized_dataset.append(normalize_pixel_values(dataset[i]))

    return np.array(normalized_dataset)
