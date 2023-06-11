from torchvision.models import vgg19_bn, VGG19_BN_Weights
import torch
import torch.nn as nn

class VGG_19_BN(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        transform = VGG19_BN_Weights.DEFAULT.transforms(antialias=True)

        bias = model.features[0].bias
        # take average of RGB weights
        new_weight = torch.unsqueeze(model.features[0].weight.sum(dim=1) / 3, 1)
        new_layer = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        new_layer.weight = nn.Parameter(new_weight)
        new_layer.bias = nn.Parameter(bias)
        # swap old layer with new layer
        model.features[0] = new_layer

        transform.mean = [sum(transform.mean) / len(transform.mean)]
        transform.std = [sum(transform.std) / len(transform.std)]

        for p in model.parameters():
            p.requires_grad_(False)

        # remove last layer keep ton of features
        #model.classifier = nn.Identity()
        model.classifier = model.classifier[:1]

        model.eval()

        self.dtype = dtype
        self.model = model
        self.transform = transform
        self.number_of_features = 4096

    def forward(self, x):
        return self.model(self.transform(x).type(self.dtype))

