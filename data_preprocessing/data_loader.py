from keras import datasets as ds
import data_normalization as dn
import numpy as np
import matplotlib.pyplot as plt
import cv2 

def load_mnist() -> dict:
    """Loads normalized mnist dataset

    Returns:
        dict: {x_train, y_train, x_test, y_test}
    """
    
    (x_train, y_train), (x_test, y_test) = ds.mnist.load_data()

    normalized_train_dataset = np.array(dn.normalize_mnist_dataset(dataset=x_train))
    normalized_test_dataset = np.array(dn.normalize_mnist_dataset(dataset=x_test))
    
    return {
        'x_train': normalized_train_dataset,
        'y_train': y_train,
        'x_test': normalized_test_dataset,
        'y_test': y_test
    }

def load_image(path:str) -> np.ndarray:
    """Loads image of a given path, normalizes it and returns

    Args:
        path (str): \timage path

    Returns:
        np.ndarray: normalized for mnist image
    """
    
    img = cv2.imread(path)
    img = dn.normalize(img)
    return img