import os
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data-path", required=True, help="Path to the eval data")
parser.add_argument("--mask-path", required=True, help="Path to the mask data")
parser.add_argument("--save-path", required=True, help="Path to save the data")
parser.add_argument(
    "--threshold", type=float, default=0.3, help="Threshold for background"
)


def main(args):
    DATA_PATH = args.data_path
    MASK_PATH = args.mask_path
    SAVE_PATH = args.save_path
    THRESHOLD = args.threshold

    classes = os.listdir(MASK_PATH)
    classes.sort()

    varieties = [
        # "fg_mask",
        # "mixed_next",
        # "mixed_same",
        # "mixed_rand",
        # "only_fg",
        # "original",
        "no_fg",
        "only_bg_b",
        "only_bg_t",
    ]

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for variety in varieties:
        if not os.path.exists(os.path.join(SAVE_PATH, variety)):
            os.makedirs(os.path.join(SAVE_PATH, variety))
        for class_name in classes:
            if not os.path.exists(os.path.join(SAVE_PATH, variety, class_name)):
                os.makedirs(os.path.join(SAVE_PATH, variety, class_name))
    for class_name in classes:
        class_path = os.path.join(MASK_PATH, class_name)
        for mask in os.listdir(class_path):
            mask_path = os.path.join(class_path, mask)
            img_mask = np.load(mask_path)
            total_pixels = img_mask.shape[0] * img_mask.shape[1]
            fg_pixels = np.sum(img_mask)
            fg_ratio = fg_pixels / total_pixels
            bg_ratio = 1 - fg_ratio
            if bg_ratio > THRESHOLD:
                for variety in varieties:
                    data_path = os.path.join(DATA_PATH, variety, "val", class_name)
                    img_name = mask.split(".")[0] + ".JPEG"
                    if variety.startswith("mixed"):
                        img_list = os.listdir(data_path)
                        for img in img_list:
                            temp = img_name.split(".")[0]
                            if img.startswith(f"fg_{temp}"):
                                img_name = img
                                break
                    elif variety == "fg_mask":
                        img_name = mask
                    os.system(
                        "cp {} {}".format(
                            os.path.join(data_path, img_name),
                            os.path.join(SAVE_PATH, variety, class_name, img_name),
                        )
                    )
            else:
                pass


if __name__ == "__main__":
    main(parser.parse_args())
