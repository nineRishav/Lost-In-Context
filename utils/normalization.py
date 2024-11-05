import numpy as np
import torch.nn as nn
import torch

def min_max_normalize(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be of `np.ndarray` type")
    
    if np.min(image) == np.max(image):
        return image
    else:
        return (image - np.min(image)) / (np.max(image) - np.min(image)) 

def standard_normalize(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be of `np.ndarray` type")
    
    if np.std(image) == 0:
        return image
    else:
        return (image - np.mean(image)) / np.std(image)

def ReLU_min_max_normalize(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("Image should be of `np.ndarray` type")
    
    image = torch.tensor(image)
    output = nn.functional.relu(image)
    output = output.numpy()
    
    if np.min(output) == np.max(output):
        return output
    else:
        return min_max_normalize(output)

