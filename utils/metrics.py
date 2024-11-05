import numpy as np
from typing import List
import utils.normalization as utils


def get_attribution(attention_map: np.ndarray, mask: np.ndarray):
    attention_map = utils.ReLU_min_max_normalize(attention_map)
    total_importance = np.sum(attention_map)
    class_attribution = np.sum(attention_map * mask)
    class_attribution_ratio = class_attribution / (total_importance + 1e-10)
    context_attribution_ratio = 1 - class_attribution_ratio
    return class_attribution_ratio, context_attribution_ratio


def get_baseline_attributions(attention_map: np.ndarray, mask: np.ndarray):
    attention_map = utils.ReLU_min_max_normalize(attention_map)
    act_shape = attention_map.shape
    total_importance = np.sum(attention_map)
    importance_per_pixel = total_importance / np.prod(act_shape)
    new_attention_map = np.ones(act_shape) * importance_per_pixel
    return get_attribution(new_attention_map, mask), importance_per_pixel


def get_pixel_attribution(attention_map: np.ndarray, mask: np.ndarray):
    attention_map = utils.ReLU_min_max_normalize(attention_map)
    total_importance = np.sum(attention_map)
    obj_attribution = np.sum(attention_map * mask)
    context_attribution = total_importance - obj_attribution

    total_pixels = np.prod(attention_map.shape)
    obj_pixels = np.sum(mask)
    context_pixels = total_pixels - obj_pixels

    obj_attribution_per_pixel = obj_attribution / obj_pixels
    context_attribution_per_pixel = context_attribution / context_pixels

    return obj_attribution_per_pixel, context_attribution_per_pixel

def get_zero_pixel_attribution(attention_map: np.ndarray, mask: np.ndarray):
    total_importance = np.sum(attention_map)
    obj_attribution = np.sum(attention_map * mask)
    context_attribution = total_importance - obj_attribution

    zero_pixel_obj = np.sum((attention_map * mask) == 0)
    zero_pixel_context = np.sum((attention_map * (1 - mask)) == 0)

    total_pixels = np.prod(attention_map.shape)
    obj_pixels = zero_pixel_obj - np.sum(1 - mask)
    context_pixels = zero_pixel_context - np.sum(mask)

    obj_attribution_per_pixel = obj_attribution / (obj_pixels + 1)
    context_attribution_per_pixel = context_attribution / (context_pixels + 1)
    per_zero_pixel_obj = obj_pixels / np.sum(mask)
    per_zero_pixel_context = context_pixels / np.sum(1 - mask)

    return (
        obj_attribution_per_pixel,
        context_attribution_per_pixel,
        obj_pixels,
        context_pixels,
        total_pixels,
        per_zero_pixel_obj,
        per_zero_pixel_context,
    )


def get_non_zero_pixel_attribution(attention_map: np.ndarray, mask: np.ndarray):
    attention_map = utils.ReLU_min_max_normalize(attention_map)
    total_importance = np.sum(attention_map)
    obj_attribution = np.sum(attention_map * mask)
    context_attribution = total_importance - obj_attribution

    zero_pixel_obj = np.sum((attention_map * mask) == 0)
    zero_pixel_context = np.sum((attention_map * (1 - mask)) == 0)

    total_pixels = np.prod(attention_map.shape)
    
    count_zeroes_obj_pixels = zero_pixel_obj - np.sum(1 - mask)
    count_non_zeroes_obj_pixels = np.sum(mask) - count_zeroes_obj_pixels           #new lines for object pixels
    
    count_zeroes_context_pixels = zero_pixel_context - np.sum(mask)
    count_non_zeroes_context_pixels = np.sum(1- mask) - count_zeroes_context_pixels           #new lines for context pixels
    
    obj_attribution_per_non_zeroes_pixel = obj_attribution / (count_non_zeroes_obj_pixels + 1)
    obj_attribution_per_all_pixel = obj_attribution / (np.sum(mask))

    context_attribution_per_non_zeroes_pixel = context_attribution / (count_non_zeroes_context_pixels + 1)
    context_attribution_per_all_pixel = context_attribution / (np.sum(1 - mask))

    # percentage of obj_attribution
    per_obj_attribution = obj_attribution / total_importance
    per_context_attribution = context_attribution / total_importance

    non_zero_pixel_obj_per_pixel = count_non_zeroes_obj_pixels / np.sum(mask)
    non_zero_pixel_context_per_pixel = count_non_zeroes_context_pixels / np.sum(1 - mask)

    return (
        obj_attribution_per_non_zeroes_pixel,
        context_attribution_per_non_zeroes_pixel,
        obj_attribution_per_all_pixel,
        context_attribution_per_all_pixel,
        
        obj_attribution,
        context_attribution,
        total_importance,
        per_obj_attribution,
        per_context_attribution,
        100,
        count_non_zeroes_obj_pixels,
        count_non_zeroes_context_pixels,
        non_zero_pixel_obj_per_pixel,
        non_zero_pixel_context_per_pixel,
    )


def get_cdf(prob_dist: np.ndarray):
    return np.cumsum(prob_dist)


def optimal_dist_change(softmax1: np.ndarray, softmax2: np.ndarray):
    cdf1 = get_cdf(softmax1)
    cdf2 = get_cdf(softmax2)
    return np.sum(np.abs(cdf1 - cdf2))


def optimal_dist_change_list(softmax1: np.ndarray, softmax2: List[np.ndarray]):
    out = []
    for softmax in softmax2:
        out.append(optimal_dist_change(softmax1, softmax))
    return out
