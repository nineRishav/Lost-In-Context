import os
import numpy as np
import json
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--data-path", type=str, default="/DATA1/konda/XAI/data/imagenet-9/bg_challenge/fg_mask/val")

def convert_numpy_to_python_standard_types(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, list):
        return [convert_numpy_to_python_standard_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python_standard_types(value) for key, value in obj.items()}
    else:
        return str(obj)

def calculate_threshold_statistics(args):
    data_directory_path = args.data_path
    Bigger_Images_Range = [0.3, 0.5]
    Smaller_Category_Range = [0.80, 1]
    Middle_Category_Range = [0.5, 0.8]  # New range added

    class_names = os.listdir(data_directory_path)
    class_names.sort()
    
    if not os.path.exists("./../results/outputs-2nd-innings/thresholder"):         
        os.makedirs("./../results/outputs-2nd-innings/thresholder")
    # output_file = open("./../results/outputs-2nd-innings/thresholder/context-bigger-and-smaller.txt", "w")
    
    total_bigger_images = 0
    total_smaller_category = 0
    total_middle_category = 0  # Initialize total count
    
    for i, class_name in enumerate(class_names):
        class_directory_path = os.path.join(data_directory_path, class_name)
        bigger_images_count = 0  # Reset counts for each class
        smaller_category_count = 0
        middle_category_count = 0  # Initialize new count
        # output_file.write(f"{i+1}. Class: {class_name}\n")
        for mask_file in os.listdir(class_directory_path):
            mask_file_path = os.path.join(class_directory_path, mask_file)
            image_mask = np.load(mask_file_path)
            total_pixels = image_mask.shape[0] * image_mask.shape[1]
            foreground_pixels = np.sum(image_mask)
            foreground_ratio = foreground_pixels / total_pixels
            background_ratio = 1 - foreground_ratio
            if Bigger_Images_Range[0] <= background_ratio <= Bigger_Images_Range[1]:
                bigger_images_count += 1
            elif Smaller_Category_Range[0] <= background_ratio <= Smaller_Category_Range[1]:
                smaller_category_count += 1
            elif Middle_Category_Range[0] <= background_ratio <= Middle_Category_Range[1]:  # Check for new category
                middle_category_count += 1
        # output_file.write(f"Bigger Images: {bigger_images_count}\n")
        # output_file.write(f"Middle Category: {middle_category_count}\n")  # Write new category count
        # output_file.write(f"Smaller Category: {smaller_category_count}\n")
        # output_file.write("#########################################\n\n\n")
        
        # Update total counts for each class
        total_bigger_images += bigger_images_count
        total_smaller_category += smaller_category_count
        total_middle_category += middle_category_count
    
    # output_file.close()

    category_statistics = {
        "Bigger Images": total_bigger_images,
        "Middle Category": total_middle_category,
        "Smaller Category": total_smaller_category
    }
    category_statistics = convert_numpy_to_python_standard_types(category_statistics)
    
    json.dump(category_statistics, open("./../results/outputs-2nd-innings/thresholder/context-bigger-and-smaller.json", "w"))


if __name__ == "__main__":
    calculate_threshold_statistics(arg_parser.parse_args())
