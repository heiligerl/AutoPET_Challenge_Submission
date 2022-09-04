from monai import transforms
import numpy as np

train_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=['ct', 'suv', 'mask']),  # 'ct', 'suv', 'mask'
        transforms.AddChanneld(keys=['ct', 'suv', 'mask']),
        transforms.Orientationd(keys=['ct', 'suv', 'mask'], axcodes="LAS"),
        transforms.NormalizeIntensityd(
            keys=['suv'],
            nonzero=True,
        ),
        transforms.ScaleIntensityRangePercentilesd(
            keys=['ct'],
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        transforms.CropForegroundd(keys=['ct', 'suv', 'mask'], source_key='ct'),
        transforms.ConcatItemsd(keys=['ct', 'suv'], name='petct', dim=0),
        transforms.RandCropByPosNegLabeld(
            keys=['petct', 'mask'],
            label_key='mask',
            spatial_size=(96, 96, 96),
            pos=2,
            neg=1,
            num_samples=4,
            image_key='petct',
            image_threshold=0,
        ),
        transforms.RandRotated(
            keys=['petct', 'mask'],
            range_x= (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            range_y= (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            range_z= (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            prob=0.2,
            mode=['bilinear', 'nearest'],
        ),
        transforms.ToTensord(keys=['petct', 'mask']),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=['ct', 'suv', 'mask']),
        transforms.AddChanneld(keys=['ct', 'suv', 'mask']),
        transforms.Orientationd(keys=['ct', 'suv', 'mask'], axcodes="LAS"),
        transforms.NormalizeIntensityd(
            keys=['suv'],
            nonzero=True,
        ),
        transforms.ScaleIntensityRangePercentilesd(
            keys=['ct'],
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        transforms.CropForegroundd(keys=['ct', 'suv', 'mask'], source_key='ct'),
        transforms.ConcatItemsd(keys=['ct', 'suv'], name='petct', dim=0),
        transforms.ToTensord(keys=['petct', 'mask']),  
    ]
)
