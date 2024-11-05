import os
import re
import numpy as np
import json
import argparse
from tqdm import tqdm
from utils.metrics import *
import time
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import utils.get_model as utils


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return str(obj)


def write_stats_to_file(file, stats, correct_class, predicted_class_1k, predicted_class_10, img_name):
    file.write(f"{img_name}:\n")
    file.write(f"Object Att. per Non-Zero Pixel: {round(stats[0], 3)}\n")
    file.write(f"Context Att. per Non-Zero Pixel: {round(stats[1], 3)}\n")
    file.write(f"Object Att. per All Pixel: {round(stats[2], 3)}\n")
    file.write(f"Context Att. per All Pixel: {round(stats[3], 3)}\n")
    file.write(f"Obj Attribution: {round(stats[4], 3)}\n")
    file.write(f"Context Attribution: {round(stats[5], 3)}\n")
    file.write(f"Total Importance: {round(stats[6], 3)}\n")
    file.write(f"Non-Zero Object Pixels: {round(stats[7], 3)}\n")
    file.write(f"Non-Zero Context Pixels: {round(stats[8], 3)}\n")
    file.write(f"Non-Zero Pixels % in Object: {round(stats[9], 3)}\n")
    file.write(f"Non-Zero Pixels % in Context: {round(stats[10], 3)}\n")
    file.write(f"Correct Class: {correct_class}\n")
    file.write(f"Predicted Class (1k): {predicted_class_1k}\n")
    file.write(f"Predicted Class (10 class): {predicted_class_10}\n")
    file.write("\n")


def transform_image(image_path):
    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # default values for imagenet
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    input_image = Image.open(image_path).convert('RGB')
    return image_transform(input_image).unsqueeze(0)


def transform_image_no_normalization(image_path):
    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    input_image = Image.open(image_path).convert('RGB')
    return image_transform(input_image).unsqueeze(0)


def main(args):
    class_10_mapping = {}
    with open("in_to_in9.json", "r") as f:
        class_10_mapping.update(json.load(f))

    class_1000_mapping = {}
    with open("imagenet_class_index.json","r",) as f:
        class_1000_mapping.update(json.load(f))

    MASK_PATH = args.mask_path
    DATA_BASE_PATH = args.data_base_path
    OUTPUT_PATH = args.output_path
    WRONG_CLASS_PATH_JSON = args.wrong_class_path_json

    VARIETIES = ["original", "only_fg", "mixed_next", "mixed_rand", "mixed_same"]
    METHODS = [
        "GradCAM",
        "GradCAMPlusPlus",
        "FullGrad",
        "GuidedBackpropReLUModel",
        "ScoreCAM",
    ]

    output_path_directory = os.path.join(OUTPUT_PATH, "metric")
    if not os.path.exists(output_path_directory):
        os.makedirs(output_path_directory)

    output_txt_file_path = os.path.join(
        output_path_directory, f"Non_Zero_Pixel_Counts-{TYPE}.txt"
    )
    output_json_file_path = os.path.join(
        output_path_directory, f"Non_Zero_Pixel_Counts-{TYPE}.json"
    )

    output_file = open(output_txt_file_path, "w")
    results = {}

    with open(WRONG_CLASS_PATH_JSON) as f:
        wrong_data = json.load(f)

    # Load model
    model = utils.get_model(MODEL_NAME)
    model.eval()
    model.cuda()

    classes = os.listdir(MASK_PATH)
    classes.sort()
    # classes = ["05_insect"]

    for method in METHODS:
        results[method] = {}
        output_file.write(f"{method}:\n")
        print(f"{method}:")
        variety = VARIETY
        results[method][variety] = {}
        output_file.write("###########################\n")
        output_file.write(f"\t{variety}:\n")
        print(f"{variety}:")
        output_file.write("###########################\n")
        data_set_stats = np.zeros(11, dtype=np.float32)
        dataset_count = 0
        for class_name in classes:
            for img_name in os.listdir(os.path.join(DATA_BASE_PATH, variety, class_name)):
                if variety == "original" or variety == "only_fg":
                    req_img_name = img_name
                else:
                    match = re.search(r"n\d+_\d+", img_name)
                    if match:
                        req_img_name = match.group(0)

                mask = np.load(os.path.join(MASK_PATH, class_name, req_img_name + ".npy"))

                all_images = wrong_data[class_name]["wrong_pred_paths"]
                image_list = [image_set[0] for image_set in all_images if image_set]

                if img_name in image_list:
                    activation_path = os.path.join(
                        DATA_BASE_PATH,
                        variety,
                        class_name,
                        img_name,
                        method + ".npy",
                    )

                    if not os.path.exists(activation_path):
                        print(f"Activation map not found for {variety} {class_name} {img_name} {method}")
                        continue

                    atten = np.load(activation_path)

                    if method == "ScoreCAM":
                        pass
                    elif method == "GuidedBackpropReLUModel":
                        atten = np.average(atten, axis=2)
                    else:
                        atten = atten.squeeze(0)
                    pix_stats = np.array(get_non_zero_pixel_attribution(atten, mask))
                    if np.isnan(pix_stats).any():
                        print(f"{list(pix_stats)} : {variety} {class_name} {img_name} {method}")

                    # Get prediction
                    source_path = "/DATA1/konda/XAI/data/imagenet-9/data_context_0.3"
                    img_name_jpeg = img_name+".JPEG"
                    img_path = os.path.join(source_path, variety, class_name, img_name_jpeg)
                    
                    if MODEL_NAME == "resnet50_in9l":
                        img = transform_image_no_normalization(img_path).cuda()
                    else:
                        img = transform_image(img_path).cuda()

                    with torch.no_grad():
                        output = model(img)
                        output = nn.Softmax(dim=1)(output)
                        _, prediction = torch.max(output, 1)
                        predicted_class = prediction.cpu().numpy()[0]
                        prediction_1k = class_1000_mapping[str(predicted_class)][1]          # Gives 1000 class prediction name for storing it in json file
                        # name of the class
                        predicted_10 = class_10_mapping[str(predicted_class)]              # Gives 10 class prediction
                    
                    correct_class = class_name
                    pix_stats_with_classes = np.append(pix_stats, [correct_class, prediction_1k, predicted_10])
                    data_set_stats += pix_stats[:11]
                    dataset_count += 1

                    results[method][variety][img_name] = [
                        round(num, 3) if isinstance(num, (int, float)) else num for num in pix_stats_with_classes
                    ]
                    write_stats_to_file(output_file, pix_stats, correct_class, prediction_1k, predicted_10, img_name)

        data_set_stats /= dataset_count

        results[method][variety]["variant_stats"] = [
            str(round(num, 3)) for num in data_set_stats
        ]
        output_file.write("-----------------------------------------\n")
        print("-----------------------------------------")
        output_file.write("Variant Stats:\n")
        print("Variant Stats:")
        
        write_stats_to_file(output_file, data_set_stats, "N/A", "N/A", "N/A", "variant_stats")
        output_file.write("\n\n")
        print("\n\n")

    output_file.close()
    results = convert_to_serializable(results)

    with open(output_json_file_path, "w") as f:
        json.dump(results, f, indent=4, cls=MyEncoder)

    print("JSON and txt Saved to directory: ", output_path_directory)


if __name__ == "__main__":
    MODEL_NAME = "resnet50"
    TYPES = [
        "wrong-mixed_next",
        "wrong-mixed_rand",
        "wrong-mixed_same",
        "wrong-only_fg",
        "wrong-original",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask_path",
        type=str,
        default="/DATA1/konda/XAI/data/imagenet-9/bg_challenge/fg_mask/val",
    )
    parser.add_argument(
        "--data_base_path",
        type=str,
        default=f"/DATA1/konda/XAI/data/imagenet-9/activation-maps/activations-{MODEL_NAME}",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=f"/DATA1/konda/XAI/XAI-Project/imagenet_9_exp/results/outputs-2nd-innings/{MODEL_NAME}/paper_new",
    )
    args = parser.parse_args()

    start_time = time.time()
    for TYPE in TYPES:
        VARIETY = TYPE.split("-")[1]
        args.wrong_class_path_json = f"/DATA1/konda/XAI/XAI-Project/imagenet_9_exp/results/outputs-final-innings/{MODEL_NAME}/paths/accuracy-image_paths-wrong-ooc-{VARIETY}-A2.json"
        main(args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        "Total Time Taken for {}: {:02d}:{:02d}:{:02d}".format(
            VARIETY,
            int(elapsed_time // 3600),
            int((elapsed_time % 3600) // 60),
            int(elapsed_time % 60),
        )
    )
