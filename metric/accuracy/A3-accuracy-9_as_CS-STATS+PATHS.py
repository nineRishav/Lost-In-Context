from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
from argparse import ArgumentParser, Namespace
from PIL import Image
from tqdm import tqdm
import utils.get_model as utils

arg_parser = ArgumentParser()
arg_parser.add_argument(
    "--arch",
    default="vit_base",
    help="Model architecture, if loading a model checkpoint.",
)
arg_parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint.")
arg_parser.add_argument(
    "--data-path",
    default=f"../../data/imagenet-9/9_as_cs/Context",
    help="Path to the eval data",
)


def transform_image(image_path):
    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],  # default values for imagenet
            #     std=[0.229, 0.224, 0.225],
            # ),
        ]
    )
    input_image = Image.open(image_path)
    return image_transform(input_image).unsqueeze(0)

def transform_image_no_normalization(image_path):
    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    input_image = Image.open(image_path)
    return image_transform(input_image).unsqueeze(0)

def main(args: Namespace):
    class_10_mapping = {}
    with open("in_to_in9.json", "r") as f:
        class_10_mapping.update(json.load(f))

    class_1000_mapping = {}
    with open(
        "imagenet_class_index.json",
        "r",
    ) as f:
        class_1000_mapping.update(json.load(f))

    # Load model
    model_name = args.arch
    model = utils.get_model(model_name)
    model.eval()
    model.cuda()

    # Open Output File
    # Define base output path
    output_base_path = f"./results/outputs-final-innings/{model_name}/accuracy"
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    # Define output file paths
    output_json_file_path = os.path.join(
        output_base_path, f"A3-accuracy-all-variant-as_CS-nonorm.json"
    )
    output_json_images_file_path = os.path.join(output_base_path, f"paths")
    output_json_images_file_path = os.path.join(
        output_base_path, f"A3-correct-imagepaths-among_all_variants-as_CS-nonorm.json"
    )

    selected_variants = [
        "fog",
        "gaussian_noise",
        "motion_blur",
        "pixelate",
        "snow",
    ]

    DATA_BASE_PATH = args.data_path
    data_variants = [
        variant
        for variant in os.listdir(DATA_BASE_PATH)
        if variant in selected_variants
    ]

    data_variants.sort(reverse=True)
    variant_stats = {}
    level = "4"
    for data_variant in data_variants:
        print(f"Processing data variant: {data_variant}")

        class_statistics = {}
        wrong_pred_classes = {}
        image_paths = {}

        class_names = os.listdir(os.path.join(DATA_BASE_PATH, data_variant, level))
        class_names.sort()
        wrong_pred_classes = {class_name: set() for class_name in class_names}
        class_accuracy = []
        out_of_class_count_each_class = 0
        out_of_class_count = 0
        correct_predictions = 0
        total_predictions = 0
        for i, class_name in enumerate(class_names):
            print(f"Processing {class_name}")
            class_path = os.path.join(DATA_BASE_PATH, data_variant, level, class_name)
            image_files = os.listdir(class_path)
            image_files.sort()
            image_paths[class_name] = {"out_of_class_paths": [], "wrong_pred_paths": []}

            correct_count = 0
            out_of_class_count_each_class = 0
            total_images = len(image_files)
            total_predictions += total_images
            for image_file in tqdm(
                image_files, desc=f"Processing Images of Class: {class_name}"
            ):
                image_path = os.path.join(class_path, image_file)
                
                if model_name == "resnet50_in9l":
                    img = transform_image_no_normalization(image_path).cuda()
                else:
                    img = transform_image(image_path).cuda()

                with torch.no_grad():
                    output = model(img)
                    output = nn.Softmax(dim=1)(output)
                    _, prediction = torch.max(output, 1)
                    prediction = prediction.cpu().numpy()[
                        0
                    ]  # Gives 1000 class prediction

                    prediction_1k = class_1000_mapping[str(prediction)][
                        1
                    ]  # Gives 1000 class prediction name for storing it in json file
                    if model_name != "resnet50_in9l":
                        prediction = class_10_mapping[
                            str(prediction)
                        ]  # Gives 10 class prediction

                    if prediction != i:
                        if prediction == -1:
                            image_paths[class_name]["out_of_class_paths"].append(
                                [image_file, prediction_1k]
                            )
                            out_of_class_count_each_class += 1
                            out_of_class_count += 1
                        else:
                            image_paths[class_name]["wrong_pred_paths"].append(
                                [image_file, prediction_1k]
                            )
                            if class_name in wrong_pred_classes:
                                wrong_pred_classes[class_name].add(prediction)
                            else:
                                wrong_pred_classes[class_name] = {prediction}

                    if prediction == i:
                        correct_count += 1
                        correct_predictions += 1

            accuracy = round(correct_count / total_images, 3)
            class_accuracy.append(accuracy)
            print(
                f"{accuracy}, {total_images}(T), {correct_count}(C), {out_of_class_count_each_class}(-1), {total_images - correct_count - out_of_class_count_each_class}(W) \n"
            )
            class_statistics[class_name] = [
                accuracy,
                total_images,
                correct_count,
                out_of_class_count_each_class,
                total_images - correct_count - out_of_class_count_each_class,
            ]

        class_statistics["overall"] = [
            round(correct_predictions / total_predictions, 3),
            correct_predictions,
            total_predictions,
            out_of_class_count,
            total_predictions - correct_predictions - out_of_class_count,
        ]
        for class_name, wrong_predictions in wrong_pred_classes.items():
            if class_name in class_statistics:
                class_statistics[class_name].append(len(wrong_predictions))
            else:
                class_statistics[class_name] = [0, 0, 0, 0, 0, len(wrong_predictions)]

        # class_statistics["image_paths"] = image_paths
        variant_stats[data_variant] = class_statistics

    try:
        with open(output_json_file_path, "w") as f:
            json.dump(variant_stats, f)
    except Exception as e:
        print(f"Error saving variant_stats: {e}")

    print(f"Json file saved at : {output_json_file_path}")

    # json.dump(
    #     image_paths,
    #     open(
    #         output_json_images_file_path,
    #         "w",
    #     ),
    # )
    # print(output_json_file_path)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        "Total Time Taken: {:02d}:{:02d}:{:02d}".format(
            int(elapsed_time // 3600),
            int((elapsed_time % 3600) // 60),
            int(elapsed_time % 60),
        )
    )


"""
Work : Calculates the accuracy of all variant of the model, does not save their paths in a json file.

Things to change:
1. Change the model name  
2. Deactivate Normalization for the resnet50_in9l model ⚠️
3. Change the output path
4. Change the data path 

"""
