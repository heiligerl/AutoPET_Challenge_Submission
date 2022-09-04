import torch
import os
import cv2
import torch.nn as nn
import torch.nn.functional as nnf
from monai.transforms import AddChannel
import numpy as np
from scipy import ndimage
from models.resnet import ResNet



class Ensemble(nn.Module):
    def __init__(
        self,
        resnet18_x_fn = './weights/classifier/resnet18_x/',
        resnet18_y_fn = './weights/classifier/resnet18_y/',
        resnet50_x_fn = './weights/classifier/resnet50_x/',
        resnet50_y_fn = './weights/classifier/resnet50_y/',
        resnet18_x_debrain_fn = './weights/classifier/resnet18_x_debrain/',
        resnet18_y_debrain_fn = './weights/classifier/resnet18_y_debrain/',
        resnet50_x_debrain_fn = './weights/classifier/resnet50_x_debrain/',
        resnet50_y_debrain_fn = './weights/classifier/resnet50_y_debrain/',
        device=None) -> None:

        super().__init__()

        self.device = device
        self.resnet18_x =  ResNet(resnet_v='resnet18', in_channels=1).to(self.device)
        self.resnet18_y =  ResNet(resnet_v='resnet18', in_channels=1).to(self.device)
        self.resnet50_x =  ResNet(resnet_v='resnet50', in_channels=1).to(self.device)
        self.resnet50_y =  ResNet(resnet_v='resnet50', in_channels=1).to(self.device)
        self.resnet18_x_debrain =  ResNet(resnet_v='resnet18', in_channels=1).to(self.device)
        self.resnet18_y_debrain =  ResNet(resnet_v='resnet18', in_channels=1).to(self.device)
        self.resnet50_x_debrain =  ResNet(resnet_v='resnet50', in_channels=1).to(self.device)
        self.resnet50_y_debrain =  ResNet(resnet_v='resnet50', in_channels=1).to(self.device)


        self.resnet18_x.load_pretrained_unequal(resnet18_x_fn, best_val=True)
        self.resnet18_y.load_pretrained_unequal(resnet18_y_fn, best_val=True)
        self.resnet50_x.load_pretrained_unequal(resnet50_x_fn, best_val=True)
        self.resnet50_y.load_pretrained_unequal(resnet50_y_fn, best_val=True)
        self.resnet18_x_debrain.load_pretrained_unequal(resnet18_x_debrain_fn, best_val=True)
        self.resnet18_y_debrain.load_pretrained_unequal(resnet18_y_debrain_fn, best_val=True)
        self.resnet50_x_debrain.load_pretrained_unequal(resnet50_x_debrain_fn, best_val=True)
        self.resnet50_y_debrain.load_pretrained_unequal(resnet50_y_debrain_fn, best_val=True)



    # Expects a tensor of size (B, C, W, H, D)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mip_x, mip_y = self.gen_mip_with_brain(x)
        # Unsqueeze to add a dummy channel dimesion
        mip_x_debrain = self.gen_mip_without_brain(0, 400, 350, 400, mip_x).unsqueeze(1)
        mip_y_debrain = self.gen_mip_without_brain(0, 400, 350, 400, mip_y).unsqueeze(1)
        mip_x = mip_x.unsqueeze(1)
        mip_y = mip_y.unsqueeze(1)


        th = 0.3 # conservative threshold to avoid false negatives

        preds = (self.resnet18_x(mip_x) > th) * 1.0
        preds += (self.resnet18_y(mip_y) > th) * 1.0
        preds += (self.resnet50_x(mip_x) > th) * 1.0
        preds += (self.resnet50_y(mip_y) > th) * 1.0
        preds += (self.resnet18_x_debrain(mip_x_debrain) > th) * 1.0
        preds += (self.resnet18_y_debrain(mip_y_debrain) > th) * 1.0
        preds += (self.resnet50_x_debrain(mip_x_debrain) > th) * 1.0
        preds += (self.resnet50_y_debrain(mip_y_debrain) > th) * 1.0
        preds = (preds > 0) * 1.0


        return preds

    def gen_mip_with_brain(self, x): # assume x in shape (B, C, H, W, D)
        all_mip_x, all_mip_y = [], []
        add_channel = AddChannel()
        for b in range(x.shape[0]): # iterate over batch
            pet_vol = x[b][0]
            assert pet_vol.shape == (400, 400, 128)

            mip_x = add_channel(add_channel(torch.max(pet_vol, dim=0)[0]))
            mip_y = add_channel(add_channel(torch.max(pet_vol, dim=1)[0]))

            mip_x = nnf.interpolate(mip_x, size=(400, 400), mode='bicubic', align_corners=False)[0]
            mip_y = nnf.interpolate(mip_y, size=(400, 400), mode='bicubic', align_corners=False)[0]

            assert mip_x.shape == (1, 400, 400) and mip_y.shape == (1, 400, 400)
            all_mip_x.append(mip_x)
            all_mip_y.append(mip_y)
        mip_x = torch.cat(all_mip_x, dim=0)
        mip_y = torch.cat(all_mip_y, dim=0)
        assert mip_x.shape == (x.shape[0], 400, 400) and mip_y.shape == (x.shape[0], 400, 400)
        return mip_x, mip_y

    def gen_mip_without_brain(self, y_min, y_max, x_min, x_max, mip):
        all_mip = []
        add_channel = AddChannel()
        for ind in range(mip.shape[0]):
            x = mip[ind].cpu().detach().numpy()
            filter_mask = np.ones_like(x)

            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
            threshold_brain = np.percentile(x_norm, 95) # Get throshold to filter out brain

            x_th = (x_norm > threshold_brain) * (x_norm)

            labels, nb = ndimage.label((x_th > 0) * 1.0) # Get connected components above the threshold
            max_size = 0
            for i in range(nb): # background
                max_size = max([np.sum(labels == i), max_size])
            for i in range(nb):
                component = (labels == i)
                if np.sum(component) == max_size:
                    continue # bg
                elif np.sum(component[y_min:y_max, x_min:x_max]) > 0: # De-Brain
                    filter_mask -= component

            x *= (filter_mask * 1.0) # Remove the brain

            # Deterministic test-time data augmentations (somehow improve the performance due to increased variability between classifiers)
            x = cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x = cv2.flip(x, 0)
            x = np.clip(x, 0, 1)

            all_mip.append(torch.Tensor(x).unsqueeze(0).to(self.device)) # add batch dimension and append
        all_mip = torch.cat(all_mip, dim=0)
        return all_mip * 255
