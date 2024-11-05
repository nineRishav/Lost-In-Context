import json
import numpy as np

IN_TO_IN9 = json.load(open("in_to_in9.json"))
IDX_TO_LABEL = {
    "0": "Dog",
    "1": "Bird",
    "2": "Vehicle",
    "3": "Reptile",
    "4": "Carnivore",
    "5": "Insect",
    "6": "Instrument",
    "7": "Primate",
    "8": "Fish",
    "-1": "Unknown",
}


def get_labels(preds: np.ndarray):
    labels = []
    for pred in preds:
        in9_class = IN_TO_IN9[str(pred)]
        labels.append(IDX_TO_LABEL[str(in9_class)])
    return labels
