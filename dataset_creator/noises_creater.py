import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    DATA_PATH = args.data_path
    MASK_PATH = args.mask_path
    OUT_PATH = args.out_path

    classes = os.listdir(DATA_PATH)
    classes.sort()

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH, exist_ok=True)

    for cls in classes:
        if not os.path.exists(os.path.join(OUT_PATH, cls)):
            os.makedirs(os.path.join(OUT_PATH, cls), exist_ok=True)
    pbar = tqdm()
    for cls in classes:
        imgs = os.listdir(os.path.join(DATA_PATH, cls))
        imgs.sort()
        for img in imgs:
            img_path = os.path.join(DATA_PATH, cls, img)
            mask_path = os.path.join(MASK_PATH, cls, img)[:-5] + ".npy"
            image = plt.imread(img_path)
            mask = np.load(mask_path)
            mask = np.expand_dims(mask, axis=-1)

            BACK_PIXEL = np.zeros_like(image)
            BACK_PIXEL[:, :, 0] += int(0.485 * 255)
            BACK_PIXEL[:, :, 1] += int(0.456 * 255)
            BACK_PIXEL[:, :, 2] += int(0.406 * 255)

            new_img = image * mask + BACK_PIXEL * (1 - mask)
            plt.imsave(os.path.join(OUT_PATH, cls, img), new_img.astype(np.uint8))
            pbar.update(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/sayanta/datasets/OCD/imagenet-9/data_context_0.3/original",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="/data/sayanta/datasets/OCD/imagenet-9/data_context_0.3/fg_mask",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/data/sayanta/datasets/OCD/imagenet-9/noises/avg-noise",
    )
    args = parser.parse_args()
    main(args)
