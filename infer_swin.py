import os

import numpy as np
import torch
from monai import data, transforms
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
import SimpleITK

def convert_mha_to_nii(mha_input_path, nii_out_path):  #nnUNet specific
    img = SimpleITK.ReadImage(mha_input_path)
    #SimpleITK.WriteImage(img, nii_out_path, True)
    return img

def get_transforms():
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=['ct', 'suv']),
            transforms.AddChanneld(keys=['ct', 'suv']),
            transforms.Orientationd(keys=['ct', 'suv'], axcodes="LAS"),
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
            transforms.ConcatItemsd(keys=['ct', 'suv'], name='petct', dim=0),
            transforms.ToTensord(keys=['petct']),
            ]
            )
    
    return test_transform

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

def create_model(weights_path, device):
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=2,
        out_channels=2,
        feature_size=48,
        use_checkpoint=False
    )
    print('Model created.')

    model.to(device)
    print(f'Model is on device: {device}')

    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model weights successfully loaded from: {weights_path}')

    model.eval()
    return model

def predict(test_dl, model, device):
    with torch.no_grad():
        for batch in test_dl:
            
            petct = batch['petct'].to(device)
            
            print('Running Sliding Window Inference...')
            output = sliding_window_inference(petct, (96, 96, 96), 4, model)
            
            # Create Probablity Predictions for Lesion Channel
            pred_prob = output.softmax(dim=1)[0, 1, ...].cpu().numpy()

            ### START SANITY CHECK
            post_pred = transforms.AsDiscrete(argmax=True, to_onehot=2)
            outputs_list = data.decollate_batch(output)
            output_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]
            pred_binary = output_convert[0][1, ...].cpu().numpy()
            pred_prob_binary = (pred_prob > 0.5).astype(int)
            assert (pred_binary == pred_prob_binary).all()
            print('Sanity Check passed. ✅')
            ### END SANITY_CHECK

    return pred_prob

def create_weights_path(model_dir, fold):
    return os.path.join(model_dir, f'best_metric_model_fold{fold}_final.pth')

def infer_swin(test_files: list, model_dir: str, folds: list=[0, 1, 2, 3, 4], device: str='cuda:0'):
    # get transforms
    transform = get_transforms()

    # prepare data
    _, dl = prepare_data(
        test_files=test_files,
        test_transform=transform
        )
    
    n_folds = len(folds)

    for i, f in enumerate(folds):
        print(f'Running fold {i}')
        
        wp = create_weights_path(
            model_dir=model_dir,
            fold=f
            )
        
        m = create_model(
            weights_path=wp,
            device=device
            )
        
        pred = predict(
            test_dl=dl,
            model=m,
            device=device
            )
        
        if i == 0:
            running_preds = pred.copy()
        else:
            running_preds += pred
    
    final_prob_pred = running_preds / n_folds

    return final_prob_pred

if __name__=='__main__':
    TEST_FILES = [
        {
            'ct': '/projects/datashare/tio/autopet_submission/input_nifti/TCIA_001_0001.nii.gz',
            'suv': '/projects/datashare/tio/autopet_submission/input_nifti/TCIA_001_0000.nii.gz',
        }
    ]

    MODEL_DIR = '/projects/datashare/tio/autopet_submission/weights/swin_unetr'

    FOLDS = [0, 1, 2, 3, 4]

    DEVICE = 'cuda:0'

    pred_prob = infer_swin(
        test_files=TEST_FILES,
        model_dir=MODEL_DIR,
        folds=FOLDS,
        device=DEVICE,
        )

    np.save('/projects/datashare/tio/autopet_submission/output/swin_output.npy', pred_prob)