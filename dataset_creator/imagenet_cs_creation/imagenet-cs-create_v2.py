import os
import argparse
import numpy as np
from torchvision import transforms
from make_imagenet_c import *
from PIL import Image


def get_images(
    ori_img: np.ndarray,
    mask_img: np.ndarray,
    corruption_img: np.ndarray,
):
    mask_img = mask_img > 0
    mask_img = np.expand_dims(mask_img[:, :, 0], axis=-1)

    # Object is the original part of the image
    object_img = corruption_img * mask_img + ori_img * (1 - mask_img)
    # Context is the corrupted part of the image
    context_img = corruption_img * (1 - mask_img) + ori_img * mask_img
    return context_img, object_img


def main(args):
    ORIGINAL_DATA_DIR = args.original_data_dir
    SAVE_DIR = args.save_dir
    MASK_DIR = args.mask_dir

    # Load all corruption functions
    d = get_distortions_dict()

    # Load imagenet-S919 class names
    class_names = os.listdir(MASK_DIR)
    class_names.sort()

    # Create save directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(os.path.join(SAVE_DIR, "Context")):
        os.makedirs(os.path.join(SAVE_DIR, "Context"))
    if not os.path.exists(os.path.join(SAVE_DIR, "Object")):
        os.makedirs(os.path.join(SAVE_DIR, "Object"))

    SAVE_CONTEXT_DIR = os.path.join(SAVE_DIR, "Context")
    SAVE_OBJECT_DIR = os.path.join(SAVE_DIR, "Object")

    # Get all Corruption Types
    # corruption_types = list(set(d.keys()) - set(["Motion Blur", "Snow"]))
    corruption_types = list(d.keys())
    corruption_types.sort()
    for corr in corruption_types:
        corr_types = d[corr].__name__
        if not os.path.exists(os.path.join(SAVE_CONTEXT_DIR, corr_types)):
            os.makedirs(os.path.join(SAVE_CONTEXT_DIR, corr_types))
        if not os.path.exists(os.path.join(SAVE_OBJECT_DIR, corr_types)):
            os.makedirs(os.path.join(SAVE_OBJECT_DIR, corr_types))

        # Get all Corruption Levels
        corruption_levels = [str(n) for n in range(1, 6)]
        corruption_levels.sort()
        for level in corruption_levels:
            if not os.path.exists(os.path.join(SAVE_CONTEXT_DIR, corr_types, level)):
                os.makedirs(os.path.join(SAVE_CONTEXT_DIR, corr_types, level))
            if not os.path.exists(os.path.join(SAVE_OBJECT_DIR, corr_types, level)):
                os.makedirs(os.path.join(SAVE_OBJECT_DIR, corr_types, level))

            # Get all classes in the corruption level
            for c_names in class_names:
                if not os.path.exists(
                    os.path.join(SAVE_CONTEXT_DIR, corr_types, level, c_names)
                ):
                    os.makedirs(
                        os.path.join(SAVE_CONTEXT_DIR, corr_types, level, c_names)
                    )
                if not os.path.exists(
                    os.path.join(SAVE_OBJECT_DIR, corr_types, level, c_names)
                ):
                    os.makedirs(
                        os.path.join(SAVE_OBJECT_DIR, corr_types, level, c_names)
                    )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )

    for c_name in class_names:
        print("Processing class: ", c_name)
        # Get all images in the class
        class_images = os.listdir(os.path.join(MASK_DIR, c_name))
        for corr in corruption_types:
            for level in corruption_levels:
                for img_name in class_images:
                    try:
                        # Get the original image
                        ori_img = Image.open(
                            os.path.join(
                                ORIGINAL_DATA_DIR, img_name.split(".")[0] + ".JPEG"
                            )
                        ).convert("RGB")
                        ori_img = transform(ori_img)
                        # Get the mask image
                        mask_img = Image.open(os.path.join(MASK_DIR, c_name, img_name))
                        mask_img = np.array(transform(mask_img))
                        # Get the corruption image
                        corruption_img = d[corr](ori_img, int(level))
                        ori_img = np.array(ori_img)
                    except FileNotFoundError:
                        continue

                    # Get the context and object images
                    context_img, object_img = get_images(
                        ori_img, mask_img, corruption_img
                    )

                    # Save the images
                    context_img = Image.fromarray(context_img.astype(np.uint8))
                    object_img = Image.fromarray(object_img.astype(np.uint8))
                    print("Saving Context Image: ", img_name.split(".")[0])
                    context_img.save(
                        os.path.join(
                            SAVE_CONTEXT_DIR,
                            d[corr].__name__,
                            level,
                            c_name,
                            img_name.split(".")[0],
                        )
                        + ".JPEG"
                    )
                    print("Saving Object Image: ", img_name.split(".")[0])
                    object_img.save(
                        os.path.join(
                            SAVE_OBJECT_DIR,
                            d[corr].__name__,
                            level,
                            c_name,
                            img_name.split(".")[0],
                        )
                        + ".JPEG"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_data_dir",
        type=str,
        default="/data/sayanta/datasets/OCD/imagenet-1k/val",
        help="Path to original imagenet dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/sayanta/datasets/OCD/imagenet-CS",
        help="Path to save the dataset",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/data/sayanta/datasets/OCD/imagenet-S/ImageNetS919/validation-segmentation",
        help="Path to mask directory",
    )
    args = parser.parse_args()
    main(args)
