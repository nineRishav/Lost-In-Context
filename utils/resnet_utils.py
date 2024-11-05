import torch
from torchvision import transforms


class DataSet(object):
    """ """

    def __init__(self, ds_name, data_path, **kwargs):
        """ """
        required_args = ["num_classes", "mean", "std", "transform_test"]
        assert set(kwargs.keys()) == set(required_args), (
            "Missing required args, only saw %s" % kwargs.keys()
        )
        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(kwargs)


class ImageNet9(DataSet):
    """ """

    def __init__(self, data_path, **kwargs):
        """ """
        ds_name = "ImageNet9"
        ds_kwargs = {
            "num_classes": 9,
            "mean": torch.tensor([0.4717, 0.4499, 0.3837]),
            "std": torch.tensor([0.2600, 0.2516, 0.2575]),
            "transform_test": transforms.ToTensor(),
        }
        super(ImageNet9, self).__init__(ds_name, data_path, **ds_kwargs)


class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class FakeReLUM(torch.nn.Module):
    def forward(self, x):
        return FakeReLU.apply(x)


class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l - 1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input


class InputNormalize(torch.nn.Module):
    """
    A module (custom layer) for normalizing the input to have a fixed
    mean and standard deviation (user-specified).
    """

    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean) / self.new_std
        return x_normalized


class NormalizedModel(torch.nn.Module):
    """ """

    def __init__(self, model, dataset):
        super(NormalizedModel, self).__init__()
        self.normalizer = InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, inp):
        """ """
        normalized_inp = self.normalizer(inp)
        output = self.model(normalized_inp)
        return output
