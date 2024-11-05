import timm
from .resnet import resnet50
from .resnet_utils import NormalizedModel, ImageNet9
import torch

CHECKPOINT_PATH = "pretrained_models/in9l_resnet50.pt"


def get_model(model_name: str):
    if model_name == "resnet101":
        model = timm.create_model("resnet101", pretrained=True)
    elif model_name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True)
    elif model_name == "efficientnet":
        model = timm.create_model("efficientnet_b0", pretrained=True)
    elif model_name == "vit_base":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
    elif model_name == "resnet50_in9l":
        model = resnet50(pretrained=False, num_classes=9)
        dataset = ImageNet9("./temp/")
        model = NormalizedModel(model, dataset)
        weights = torch.load(CHECKPOINT_PATH)["model"]
        weights = {k[len("module.") :]: v for k, v in weights.items()}
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        model_dict.update(weights)
        model.load_state_dict(model_dict)

    elif model_name == "unknown-resnet50":
        model = timm.create_model("resnet50", pretrained=False)
    elif model_name == "unknown-resnet101":
        model = timm.create_model("resnet101", pretrained=False)
    elif model_name == "unknown-efficientnet":
        model = timm.create_model("efficientnet_b0", pretrained=False)
    elif model_name == "unknown-resnet50_in9l":
        model = resnet50(pretrained=False, num_classes=9)
        dataset = ImageNet9("./temp/")
        model = NormalizedModel(model, dataset)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    return model
