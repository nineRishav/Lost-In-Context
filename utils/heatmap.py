import numpy as np
from utils.normalization import min_max_normalize
import cv2


def generate_heatmap(
    attention_map: np.ndarray, cmap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    # Ensure that attention_map is in the [0, 1] range
    attention_map = np.clip(attention_map, 0, 1)

    # Replace NaN and infinite values with 0
    attention_map[np.isnan(attention_map) | np.isinf(attention_map)] = 0

    # Scale the values to the range [0, 255] and cast to uint8
    attention_map = (attention_map * 255).astype(np.uint8)
    # print(attention_map.shape)
    # print(attention_map)

    # Apply the colormap
    heatmap = cv2.applyColorMap(src=attention_map, colormap=cmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


def add_image_and_heatmap(
    image: np.ndarray, heatmap: np.ndarray, image_weight: float
) -> np.ndarray:
    image = min_max_normalize(image)
    heatmap = min_max_normalize(heatmap)
    out = min_max_normalize(image_weight * image + (1 - image_weight) * heatmap)
    return np.uint8(out * 255)
