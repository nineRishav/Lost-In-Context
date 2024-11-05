import os
import json
import argparse
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--data-path",
    type=str,
    default="/DATA1/konda/XAI/data/imagenet-9/bg_challenge/fg_mask/val",
)


def calculate_threshold_statistics(args):
    data_directory_path = args.data_path
    Bigger_Images_Range = [0.3, 0.5]
    Middle_Category_Range = [0.5, 0.8]
    Smaller_Category_Range = [0.80, 1]

    class_names = os.listdir(data_directory_path)
    class_names.sort()

    if not os.path.exists("./../results/outputs-2nd-innings/thresholder"):
        os.makedirs("./../results/outputs-2nd-innings/thresholder")
    output_file = open(
        "./../results/outputs-2nd-innings/thresholder/context-category-class-summary.txt",
        "w",
    )

    class_images = {}

    for i, class_name in enumerate(class_names):
        class_directory_path = os.path.join(data_directory_path, class_name)
        images_in_class = {"Bigger": [], "Middle": [], "Smaller": []}
        output_file.write(f"{i+1}. Class: {class_name}\n")
        for mask_file in os.listdir(class_directory_path):
            image_name = os.path.splitext(mask_file)[
                0
            ]  # Get the image name without the extension
            mask_file_path = os.path.join(class_directory_path, mask_file)
            image_mask = np.load(
                mask_file_path
            )  # Assuming the data is loaded using NumPy
            total_pixels = image_mask.shape[0] * image_mask.shape[1]
            foreground_pixels = np.sum(image_mask)
            foreground_ratio = foreground_pixels / total_pixels
            background_ratio = 1 - foreground_ratio

            if Bigger_Images_Range[0] <= background_ratio <= Bigger_Images_Range[1]:
                images_in_class["Bigger"].append(image_name)
            elif (
                Middle_Category_Range[0] <= background_ratio <= Middle_Category_Range[1]
            ):
                images_in_class["Middle"].append(image_name)
            elif (
                Smaller_Category_Range[0]
                <= background_ratio
                <= Smaller_Category_Range[1]
            ):
                images_in_class["Smaller"].append(image_name)

        class_images[
            class_name
        ] = images_in_class  # Add the dictionary of image names to the main dictionary
        output_file.write(f"Bigger Images: {len(images_in_class['Bigger'])}\n")
        output_file.write(f"Middle Images: {len(images_in_class['Middle'])}\n")
        output_file.write(f"Smaller Images: {len(images_in_class['Smaller'])}\n")
        output_file.write("#########################################\n\n\n")

    output_file.close()

    # Save image names to a JSON file
    json.dump(
        class_images,
        open(
            "./../results/outputs-2nd-innings/thresholder/threshold-image_path.json",
            "w",
        ),
    )


if __name__ == "__main__":
    calculate_threshold_statistics(arg_parser.parse_args())
