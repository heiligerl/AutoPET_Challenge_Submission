import os

import numpy as np
import torch
from monai import data, transforms
import SimpleITK
from models.ensemble import Ensemble
from monai.transforms import(
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    CropForegroundd,
    ScaleIntensityRanged,
    Compose
)
def get_transforms():
    val_transforms= Compose(
        [
            LoadImaged(keys=["suv"]),
            AddChanneld(keys=["suv"]),
            Spacingd(keys=["suv"], pixdim=(2.0, 2.0, 3.0), mode=("bilinear")),
            Orientationd(keys=["suv"], axcodes="LAS"),
            ScaleIntensityRanged(keys=["suv"], a_min=0, a_max=15, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["suv"], source_key="suv"),
            Resized(keys=["suv"], spatial_size=[400, 400, 128]),
            ToTensord(keys=["suv"]),
        ]
    )
    return val_transforms


def prepare_data(test_files, test_transform):
    test_ds = data.Dataset(
        data=test_files,
        transform=test_transform
    )

    assert len(test_ds) == 1

    test_dl = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return test_ds, test_dl

def create_model(device):
    model = Ensemble(device=device) # TODO change the paths to the weights
    print('Model created.')

    print(f'Model is on device: {device}') # Internally done in Ensemble()

    return model

def predict(test_dl, model, device):
    outputs = []
    inp_shape = None
    with torch.no_grad():
        for batch in test_dl:
            pet = batch['suv'].to(device)
            inp_shape = pet.shape
            out = model(pet).cpu().detach().numpy()
            outputs += list(out.flatten())
    #outputs = torch.Tensor(outputs).to(device)
    outputs = np.array(outputs)
    return outputs, inp_shape # (N , 1)


def infer_classifier(test_files: list, device: str='cuda:0'):
    # get transforms
    transform = get_transforms()

    device = torch.device(device)

    # prepare data
    _, dl = prepare_data(
        test_files=test_files,
        test_transform=transform
        )

    m = create_model(
        device=device
        )

    pred, inp_shape = predict(
        test_dl=dl,
        model=m,
        device=device
        )

    return pred, inp_shape

if __name__=='__main__':
    TEST_FILES = [
        {
            'ct': '/projects/datashare/tio/autopet_submission/input_nifti/TCIA_001_0001.nii.gz',
            'suv': '/projects/datashare/tio/autopet_submission/input_nifti/TCIA_001_0000.nii.gz',
        }
    ]

    DEVICE = 'cuda:0'
    print("Starting classifier inference...")
    pred, inp_shape = infer_classifier(
        test_files=TEST_FILES,
        device=DEVICE,
        )
    print("Prediction:", pred)
    print('Done!')
