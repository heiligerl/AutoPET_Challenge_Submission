import os
from typing import Dict

import numpy as np
import SimpleITK
import torch

from ap_baseline_nnunet_process import Autopet_baseline
from infer_swin import infer_swin
from infer_class import infer_classifier
import nibabel as nib

def convert_mha_to_nii(mha_input_path, nii_out_path):  #nnUNet specific
    img = SimpleITK.ReadImage(mha_input_path)
    SimpleITK.WriteImage(img, nii_out_path, True)

def convert_nii_to_mha(nii_input_path, mha_out_path):  #nnUNet specific
    img = SimpleITK.ReadImage(nii_input_path)
    SimpleITK.WriteImage(img, mha_out_path, True)

def create_nifti_monai(path_filename_dict: dict) -> None:
    # get paths
    input_path = path_filename_dict['input_path']
    nii_path = path_filename_dict['monai_nii_path']

    # get paths to images
    ct_mha = os.listdir(os.path.join(input_path, 'images/ct/'))[0]
    pet_mha = os.listdir(os.path.join(input_path, 'images/pet/'))[0]

    # convert to nifti
    convert_mha_to_nii(os.path.join(input_path, 'images/pet/', pet_mha),
                            os.path.join(nii_path, 'TCIA_001_0000.nii.gz'))
    convert_mha_to_nii(os.path.join(input_path, 'images/ct/', ct_mha),
                            os.path.join(nii_path, 'TCIA_001_0001.nii.gz'))

    print(f'Created nifti saved at: {nii_path}')

def create_datalist(path_filename_dict) -> list:
    nii_path = path_filename_dict['monai_nii_path']

    res = [
        {
            'ct': os.path.join(nii_path, 'TCIA_001_0001.nii.gz'),
            'suv': os.path.join(nii_path, 'TCIA_001_0000.nii.gz'),
        }
    ]

    return res


def load_inputs(
        path_filename_dict: dict
    ) -> str:

    input_path = path_filename_dict['input_path']
    nii_path = path_filename_dict['nii_path']

    ct_mha = os.listdir(os.path.join(input_path, 'images/ct/'))[0]
    pet_mha = os.listdir(os.path.join(input_path, 'images/pet/'))[0]
    uuid = os.path.splitext(ct_mha)[0]

    convert_mha_to_nii(os.path.join(input_path, 'images/pet/', pet_mha),
                            os.path.join(nii_path, 'TCIA_001_0000.nii.gz'))
    convert_mha_to_nii(os.path.join(input_path, 'images/ct/', ct_mha),
                            os.path.join(nii_path, 'TCIA_001_0001.nii.gz'))
    return uuid


def switch_nnunet_axes(nnunet_softmax: np.ndarray) -> np.ndarray:
    res = np.transpose(nnunet_softmax[1, ...], (2, 1, 0))
    return res


def get_nnunet_softmax(
        path_filename_dict: dict
    ) -> np.ndarray:

    d = path_filename_dict
    apbl = Autopet_baseline(**d)
    uuid = apbl.process()
    result_path = d['result_path']
    npz_filename = d['nii_seg_file'].replace('.nii.gz', '.npz')
    #nnunet_softmax = np.load(os.path.join(result_path, npz_filename)) <--CHANGED TO: see following line
    nnunet_softmax = np.load(os.path.join(result_path, npz_filename))['softmax']
    nnunet_softmax = switch_nnunet_axes(nnunet_softmax)
    return nnunet_softmax


def get_swin_softmax(path_filename_dict: dict) -> np.ndarray:
    # get weights path
    wp = path_filename_dict['swin_weights']
    # prepare data
    files = create_datalist(path_filename_dict=path_filename_dict)
    # compute softmax
    swin_softmax = infer_swin(
        test_files=files,
        model_dir=wp
    )
    return swin_softmax

def get_classifier_prediction(path_filename_dict: dict):
    # Create the nifti files for the whole pipeline
    create_nifti_monai(path_filename_dict)

    test_files = create_datalist(path_filename_dict)

    pred, inp_shape = infer_classifier(test_files, "cuda:0")

    return pred, inp_shape

def write_outputs(
        array: np.ndarray,
        path_filename_dict: dict,
        uuid: str
    ) -> None:
    monai_nii_path = path_filename_dict['monai_nii_path']
    output_path = path_filename_dict['output_path']
    
    #ct_nifti = nib.load('/projects/datashare/tio/autopet_submission/monai_input/TCIA_001_0001.nii.gz') 
    ct_nifti = nib.load(os.path.join(monai_nii_path,"TCIA_001_0001.nii.gz" ))
    array_nifti = nib.Nifti1Image(array, affine=ct_nifti.affine, header=ct_nifti.header)
    
    nib.save(array_nifti, 'output_nifti.nii')

    convert_nii_to_mha('output_nifti.nii', os.path.join(output_path, uuid + '.mha'))

    return None

def postprocessing(prediction):
    '''Postprocessing function. Sets 3 slides at the upper bound of an numpy array to zero or sets the whole prediction to zero if the sum is smaller or equal to 10.
        Expected input shape (B,C,X,Y,Z)'''
    new_pred = np.zeros_like(prediction)
    #assert prediction.shape[-2]==prediction.shape[-3]
    #assert prediction.shape[1] == 1
    if prediction.sum() <= 10:
        return new_pred
    else:
        ## 3 slides are taken away because it has given the best results on the validation sets. 
        up = prediction.shape[-1]-3
        new_pred = np.zeros_like(prediction)
        new_pred[:,:,:up] = prediction[:,:,:up]
        return new_pred


def run_inference(path_filename_dict: dict) -> None:

    uuid = load_inputs(path_filename_dict)

    # Classifier components
    # clf_pred, array = get_classifier_prediction(path_filename_dict) <<<<<<<<<<<< Zdravko
    '''
        Binary outputs of the classifier ensemble - 1 means TUMOR, 0 means healthy ===> Directly multiply seg_mask with the clasifier output.
    '''
    clf_pred, inp_shape = get_classifier_prediction(path_filename_dict)
    if clf_pred[0] == 0:
        res = np.zeros(inp_shape)
    else:
        nnunet_softmax = get_nnunet_softmax(path_filename_dict)
        swin_softmax = get_swin_softmax(path_filename_dict)  # (X, Y, Z)
        
        avg_softmax = (nnunet_softmax + swin_softmax) / 2
        avg_binary = (avg_softmax > 0.5).astype(np.int16)

        res = postprocessing(avg_binary)

    write_outputs(
        array=res,
        path_filename_dict=path_filename_dict,
        uuid=uuid,
    )

    return None




if __name__ == "__main__":
   
   
    PATH_FILENAME_DICT = {
        'input_path': '/input/',
        'output_path': '/output/images/automated-petct-lesion-segmentation',
        'nii_path': '/opt/algorithm/nnUNet_raw_data_base/nnunet_raw_data/Task501_autoPET/imagesTs',
        'monai_nii_path': '/opt/algorithm/monai_input/',
        'result_path': '/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task501_autoPET/result',
        'nii_seg_file': 'TCIA_001.nii.gz',
        'swin_weights': '/opt/algorithm/weights/swin_unetr'
    } 
    try:
        os.makedirs(PATH_FILENAME_DICT["output_path"],exist_ok=True)  
        print("directory created")
        print(os.listdir("/output"))
        for root,dirs,files in os.walk("/output"):
            print(dirs)
    except Exception as ex: 
        print(ex)
    run_inference(PATH_FILENAME_DICT)
