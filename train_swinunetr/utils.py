import pandas as pd
from typing import Dict, List
import torch
import random
import numpy as np


def get_scan_paths(df: pd.DataFrame, col: str) -> List[str]:
    return df[col].to_list()

def add_suffixes(scan_paths: list) -> List[Dict[str, str]]:
    return [{'ct': p + 'CTres.nii.gz', 'suv': p + 'SUV.nii.gz', 'mask': p + 'SEG.nii.gz'} for p in scan_paths]

def create_datalist(df: pd.DataFrame, col:str):
    scan_paths = get_scan_paths(df, col)
    return add_suffixes(scan_paths)

def set_seed(seed: int=2206) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True