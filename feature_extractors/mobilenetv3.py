from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch
import torch.nn as nn

class MobileNetV3(nn.Module):
    def __init__(self):
        super.__init__()
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        transform = MobileNet_V3_Large_Weights.DEFAULT.transforms()
        # take average of RGB weights
        new_weight = torch.unsqueeze(model.features[0][0].weight.sum(dim=1) / 3, 1)
        new_layer = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        new_layer.weight = nn.Parameter(new_weight)
        # swap old layer with new layer
        model.features[0][0] = new_layer
        #change transform
        transform.mean = [sum(transform.mean) / len(transform.mean)]
        transform.std = [sum(transform.std) / len(transform.std)]

        for p in model.parameters():
            p.requires_grad_(False)

        # remove last layer
        model.classifier = model.classifier[:2]

        model.eval()

        self.model = model
        self.transform = transform

