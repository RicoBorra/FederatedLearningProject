import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PpvPooling2D(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, input):
        # returns values between 0 and 1, mostly ones or zeros
        transformed_input = ((F.hardtanh(input, min_val=-self.threshold, max_val=+self.threshold) / self.threshold) + 1) / 2
        # average pooling on this transformed input is almost equivalent to ppv pooling, except that the transition between one and zero is smooth instead of hard
        return self.avg(transformed_input)
    

class Rocket2D(nn.Module):
    def __init__(self, h=28, w=28, ch=1, threshold=1e-7, n1=10, n2=1000, avg=False):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        self.flatten = nn.Flatten()
        self.ppv = PpvPooling2D(threshold=threshold)
        self.layer1 = nn.ModuleList([nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=3, padding=1) for _ in range(n1)])
        self.avg = avg
        self.avgpool = nn.AvgPool2d(kernel_size=(3, 3), padding=1, stride=1)

        self.last_layer = nn.ModuleList()
        for i in range(n2):
            # kernels
            kernel_rows = np.random.choice((2, 3, 4))
            kernel_cols = np.random.choice((2, 3, 4))
            # dilations
            dilation_row = np.int32(2 ** np.random.uniform(0, np.log2((h - 1) / (kernel_rows - 1))))
            dilation_col = np.int32(2 ** np.random.uniform(0, np.log2((w - 1) / (kernel_cols - 1))))
            # padding
            if np.random.randint(2) == 1:
                padding_row = ((kernel_rows - 1) * dilation_row) // 2
                padding_col = ((kernel_cols - 1) * dilation_col) // 2
            else:
                padding_row = 0
                padding_col = 0
            # create kernel
            random_kernel = nn.Conv2d(in_channels=1, 
                                    out_channels=1, 
                                    kernel_size=(kernel_rows, kernel_cols), 
                                    dilation=(dilation_row, dilation_col),
                                    padding=(padding_row, padding_col))
            # weight initialization
            nn.init.normal_(random_kernel.weight)
            # bias initialization
            nn.init.uniform_(random_kernel.bias, a=-1, b=1)
            self.last_layer.append(random_kernel)
        self.last_layer = self.last_layer
        self.number_of_features = n1*n2


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features_list = []
            for i in range(self.n1):
                for j in range(self.n2):
                    inter = self.layer1[i](x)
                    feature = self.flatten(self.ppv(self.last_layer[j](inter)))
                    features_list.append(feature)
                    if self.avg == True:
                        feature_avg = self.flatten(self.ppv(self.last_layer[j](self.avgpool(inter))))
                        features_list.append(feature_avg)
                        del feature_avg
                    del feature
                    torch.cuda.empty_cache()
            features = torch.cat(features_list, dim=-1)

        return features
