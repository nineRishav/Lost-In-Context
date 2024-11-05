import os
import re
import numpy as np
import json
import argparse
from tqdm import tqdm
from utils.metrics import *
import time

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
    # Handle other types as needed (e.g., datetime objects)
    else:
        return str(obj)

def write_stats_to_file(file, stats):
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

""" 
Order: 
    Object/Context Att. per Non-Zero Pixel (2)
    Object/Context Att. per All Pixel (2)
    Obj/Context Attribution (2)
    Total Importance (1)
    Non-Zero Object/Context Pixels (2)
    Non-Zero Pixels % in Object/Context (2)
"""


def main(args):
    MASK_PATH = args.mask_path
    DATA_BASE_PATH = args.data_base_path
    OUTPUT_PATH = args.output_path
    WRONG_CLASS_PATH_JSON = args.wrong_class_path_json

    VARIETIES = ["gaussian_noise", "white_noise"]
    METHODS = [
        "GradCAM",
        "FullGrad",
        "ScoreCAM",
    ]
    
    # Define output file paths
    output_path_directory = os.path.join(OUTPUT_PATH, "metric")
    if not os.path.exists(output_path_directory):
        os.makedirs(output_path_directory)
    
    output_txt_file_path = os.path.join(output_path_directory, f"Non_Zero_Pixel_Counts-{TYPE}.txt")
    output_json_file_path = os.path.join(output_path_directory, f"Non_Zero_Pixel_Counts-{TYPE}.json")
    
    output_file = open(output_txt_file_path, "w")
    results = {}

    with open(WRONG_CLASS_PATH_JSON) as f:
        wrong_data = json.load(f)


    classes = os.listdir(MASK_PATH)
    classes.sort()
    
    for method in METHODS:
        results[method] = {}
        output_file.write(f"{method}:" + "\n")
        print(f"{method}:")
        variety = VARIETY
        results[method][variety] = {}
        output_file.write("###########################\n")
        output_file.write(f"\t {variety}:" + "\n")
        print(f"{variety}:")
        output_file.write("###########################\n")
        data_set_stats = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        dataset_count = 0
        for class_name in classes:
            class_stats = np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            clas_count = 0
            for img_name in os.listdir(os.path.join(DATA_BASE_PATH, variety, class_name)):
                req_img_name = img_name
                
                mask = np.load(
                    os.path.join(MASK_PATH, class_name, req_img_name + ".npy")
                )

                all_images = wrong_data[class_name]["wrong_pred_paths"] + wrong_data[class_name]["out_of_class_paths"]
                
                image_list = [image_set[0] for image_set in all_images if image_set]    # take only the image name not the prediction
                
                # print(req_img_name)
                
                # print(image_list)
                # exit()

                if img_name in image_list:
                    activation_path = os.path.join(DATA_BASE_PATH,variety,class_name,img_name,method + ".npy",)
                    
                    if not os.path.exists(activation_path):             # if the activation map is not present
                        print(f"Activation map not found for {variety} {class_name} {img_name} {method}")
                        continue

                    atten = np.load(activation_path)

                    if method == "ScoreCAM":
                        pass
                    elif method == "GuidedBackpropReLUModel":
                        atten = np.average(atten, axis=2)
                    else:
                        atten = atten.squeeze(0)
                    pix_stats = np.array(
                        get_non_zero_pixel_attribution(atten, mask))
                    if np.isnan(pix_stats).any():
                        print(f"{list(pix_stats)} : {variety} {class_name} {img_name} {method}")

                    class_stats = class_stats + pix_stats
                    clas_count += 1
                    dataset_count += 1
                
            if clas_count == 0:
                results[method][variety][class_name] = [0 for num in class_stats]
                output_file.write(f"\t\t {class_name}:" + "\n")
                print(f"ZERO Class Count : {class_name} : {clas_count}")
                write_stats_to_file(output_file, class_stats)
                continue

            data_set_stats += class_stats
            class_stats /= clas_count
            
            results[method][variety][class_name] = [round(num, 3) for num in class_stats]
            output_file.write(f"\t\t {class_name}:" + "\n")
            print(f"Completed : {class_name} : {clas_count}") 
            write_stats_to_file(output_file, class_stats)

        data_set_stats /= dataset_count
        
        results[method][variety]["variant_stats"] = [str(round(num, 3)) for num in data_set_stats]
        output_file.write("-----------------------------------------\n")
        print("-----------------------------------------")
        output_file.write("Variant Stats:" + "\n")
        print("Variant Stats:")
        
        write_stats_to_file(output_file, data_set_stats)
        output_file.write("\n\n")
        print("\n\n")

    output_file.close()
    results = convert_to_serializable(results)

    with open(output_json_file_path, "w") as f:
        json.dump(results, f, indent=4, cls=MyEncoder)
    
    print("JSON and txt Saved to directory: ", output_path_directory)


if __name__ == "__main__":
    MODEL_NAME = "resnet50"
    TYPES = ["wrong-gaussian_noise", "wrong-white_noise"]

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
        default=f"/DATA1/konda/XAI/XAI-Project/imagenet_9_exp/results/outputs-2nd-innings/{MODEL_NAME}",
    )
    args = parser.parse_args()

    start_time = time.time()
    for TYPE in TYPES:
        VARIETY = TYPE.split("-")[1]
        args.wrong_class_path_json = f"/DATA1/konda/XAI/XAI-Project/imagenet_9_exp/results/outputs-2nd-innings/{MODEL_NAME}/paths/accuracy-image_paths-wrong-ooc-{VARIETY}.json"    
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


'''
Work : Calculates the metric of all misclassified noise variant of the model for variant(gaussian and white-noise), and saves their metric in a json file

Things to change:
1. Change the model name
2. Change the correct class path json 
4. Change the output path
5. Change the data path
6. Change the mask path
7. For Efficientnet, GuidedBackpropReLUModel is missing ⚠️

'''