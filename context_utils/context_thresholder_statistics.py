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
    CONTEXT_THRESHOLD = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    class_names = os.listdir(data_directory_path)
    class_names.sort()
    classwise_accuracy = np.zeros((len(class_names), len(CONTEXT_THRESHOLD)))
    total_accuracy = np.zeros((len(CONTEXT_THRESHOLD)))
    
    if not os.path.exists("./../results/outputs-2nd-innings/TESTING"):         
        os.makedirs("./../results/outputs-2nd-innings/TESTING")
    output_file = open("./../results/outputs-2nd-innings/TESTING/context_threshold_statistics.txt", "w")
    for j, threshold in enumerate(CONTEXT_THRESHOLD):
        output_file.write(f"Threshold: {threshold}\n")
        output_file.write("------------------------------------\n")
        false_counts, total_counts = [], []
        for i, class_name in enumerate(class_names):
            output_file.write(f"{i+1}. Class: {class_name}\n")
            class_directory_path = os.path.join(data_directory_path, class_name)
            true_count, false_count = 0, 0
            for mask_file in os.listdir(class_directory_path):
                mask_file_path = os.path.join(class_directory_path, mask_file)
                image_mask = np.load(mask_file_path)
                total_pixels = image_mask.shape[0] * image_mask.shape[1]
                foreground_pixels = np.sum(image_mask)
                foreground_ratio = foreground_pixels / total_pixels
                background_ratio = 1 - foreground_ratio
                true_count += 1
                if background_ratio > threshold:
                    false_count += 1
            false_counts.append(false_count)
            total_counts.append(true_count)
            output_file.write(f"Images above threshold: {false_count} out of {true_count}\n")
            output_file.write("Percentage: {:.2f}\n".format((false_count / true_count) * 100))
            output_file.write("\n")
            classwise_accuracy[i, j] = (false_count / true_count) * 100
        output_file.write("#########################################\n")
        output_file.write(f"Total images above threshold: {sum(false_counts)} out of {sum(total_counts)}\n")
        output_file.write("Percentage: {:.2f}\n".format((sum(false_counts) / sum(total_counts) * 100)))
        output_file.write("#########################################\n\n\n")
        total_accuracy[j] = sum(false_counts) / sum(total_counts) * 100
    output_file.close()
    accuracy_statistics = dict()
    accuracy_statistics["classwise"] = classwise_accuracy
    accuracy_statistics["total"] = total_accuracy
    accuracy_statistics = convert_numpy_to_python_standard_types(accuracy_statistics)
    
    json.dump(accuracy_statistics, open("./../results/outputs-2nd-innings/TESTING/context_threshold_statistics.json", "w"))


if __name__ == "__main__":
    calculate_threshold_statistics(arg_parser.parse_args())