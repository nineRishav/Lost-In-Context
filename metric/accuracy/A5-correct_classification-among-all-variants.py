import os
import json
import argparse
import time
from tqdm import tqdm
import numpy as np
from utils.get_model import get_model
from torchvision import transforms
from PIL import Image

global model_name

def convert_to_serializable(input_object):
    if isinstance(input_object, np.float32):
        return float(input_object)
    elif isinstance(input_object, np.int32):
        return int(input_object)
    elif isinstance(input_object, np.ndarray):
        return input_object.tolist()  # Convert NumPy arrays to lists
    elif isinstance(input_object, list):
        return [convert_to_serializable(item) for item in input_object]
    elif isinstance(input_object, dict):
        return {
            key: convert_to_serializable(value) for key, value in input_object.items()
        }
    else:
        return str(input_object)

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
    image = Image.open(image_path)
    return image_transform(image).unsqueeze(0)

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


class JsonEncoder(json.JSONEncoder):
    def default(self, input_object):
        if isinstance(input_object, np.float32):
            return float(input_object)
        return super().default(input_object)


def main(program_args):
    data_directory = program_args.data_path
    global save_directory
    save_directory = program_args.save_path

    model_name = program_args.model_name
    save_directory = os.path.join(program_args.save_path, model_name)
    save_directory = os.path.join(save_directory, "paths")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    in_to_in9_mapping = {}
    with open("in_to_in9.json", "r") as file:
        in_to_in9_mapping.update(json.load(file))

    varieties = ["original", "only_fg", "mixed_next", "mixed_rand", "mixed_same"]

    # Getting classes
    class_names = os.listdir(os.path.join(data_directory, varieties[0]))
    class_names.sort()

    # Load the model
    model = get_model(model_name)
    model.eval()
    model.cuda()

    class_to_images = {}
    for class_name in class_names:
        class_to_images[class_name] = []

    total_images = 0
    correctly_classified_images = 0

    for class_index, class_name in enumerate(class_names):
        # takes image name from original folder
        image_names = os.listdir(os.path.join(data_directory, varieties[0], class_name))
        image_names.sort()

        total_images += len(image_names)

        # search for the image in all the folders and check if it is classified correctly
        for image_name in tqdm(image_names, desc=f"Processing Images for Class {class_name}"):
            image_name = image_name.split(".")[0]
            is_classification_correct = True

            for variety in varieties[0:]:
                if variety.startswith("mixed"):  # checking mixed varieties
                    temp_names = os.listdir(os.path.join(data_directory, variety, class_name))
                    temp_names.sort()
                    for temp_name in temp_names:
                        if temp_name.startswith(f"fg_{image_name}"):
                            image_path = os.path.join(data_directory, variety, class_name, temp_name)  # found the image
                            break
                else:  # original & only_fg
                    image_path = os.path.join(data_directory, variety, class_name, image_name + ".JPEG")
                image = Image.open(image_path)
                
                if program_args.model_name == "resnet50_in9l":
                    image = transform_image_no_normalization(image_path)
                else:
                    image = transform_image(image_path)

                image = image.cuda()
                prediction = model(image)
                prediction = prediction.argmax(dim=1)

                if program_args.model_name == "resnet50_in9l":
                    prediction = prediction.item()
                    # print(prediction, class_index)
                else:
                    prediction = in_to_in9_mapping[str(prediction.item())]

                if prediction != class_index:  # Wrong Classification
                    if prediction == -1:  # Out of Class Prediction
                        is_classification_correct = False
                        break

            if is_classification_correct:
                correctly_classified_images += 1
                class_to_images[class_name].append(image_name)

    serializable_image_dict = convert_to_serializable(class_to_images)

    # Save the dictionary to the JSON file
    with open(os.path.join(save_directory, "A5-correct-imagepaths-among_ALL_VARIANTS.json"), "w") as json_file:
        json.dump(serializable_image_dict, json_file, cls=JsonEncoder)

    print(f"Json saved at {save_directory}")
    print(f"Total Images: {total_images} | Correctly Classified Images: {correctly_classified_images} | Accuracy: {(correctly_classified_images/total_images)*100:.2f}%")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--data_path",
        type=str,
        default="/DATA1/konda/XAI/data/imagenet-9/data_context_0.3",
    )
    argument_parser.add_argument(
        "--save_path",
        type=str,
        default=f"./results/outputs-final-innings",
    )
    argument_parser.add_argument("--model_name", type=str, default="resnet50")
    args = argument_parser.parse_args()

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
