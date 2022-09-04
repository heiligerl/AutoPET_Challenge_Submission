import torch
import os
#import torchvision
from models.resnet_backbone import resnet18, resnet50, resnet101
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(
        self,
        resnet_v = 'resnet18',
        in_channels=3
        ) -> None:

        super().__init__()
        self.resnet_v = resnet_v
        self.in_channels = in_channels

        if resnet_v == 'resnet18':
            self.resnet = resnet18(pretrained=False, dropout=False)
        elif resnet_v == 'resnet50':
            self.resnet = resnet50(pretrained=False, dropout=False)
        elif resnet_v == 'resnet101':
            self.resnet = resnet101(pretrained=False, dropout=False)
        else:
            print('[ERROR]', self.resnet, ' version of ResNet is not supported!')
            exit()
        if self.in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.linear_act = nn.Sigmoid()

        self.model = nn.Sequential(self.resnet,
                                   self.linear_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.model(x)
        return x

    def load_pretrained_unequal(self, file, best_val=False):
        # load the weight file and copy the parameters
        if best_val:
            paths = sorted([el for el in os.listdir(file) if 'best' in el])
            file = os.path.join(file, paths[-1])
        if os.path.isfile(file):
            print('Loading pre-trained weight file.')
            if "net" in torch.load(file).keys():
                weight_dict = torch.load(file)["net"]
            else:
                weight_dict = torch.load(file)
            model_dict = self.state_dict()

            for name, param in weight_dict.items():
                if name in model_dict:
                    if param.size() == model_dict[name].size():

                        model_dict[name].copy_(param)
                        #model_dict[name] = param
                    else:
                        print(
                            f' WARNING parameter size not equal. Skipping weight loading for: {name} '
                            f'File: {param.size()} Model: {model_dict[name].size()}')
                else:
                    print(f' WARNING parameter from weight file not found in model. Skipping {name}')
            print('Loaded pre-trained parameters from file.')

        else:
            raise ValueError(f"Weight file {file} does not exist")
