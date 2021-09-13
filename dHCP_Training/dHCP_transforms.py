import monai
import pandas as pd
import re
import time
import os
import shutil
import tempfile
import tqdm
from collections import Counter, OrderedDict
import random
from datetime import datetime
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from monai.transforms.inverse import InvertibleTransform
from monai.apps import DecathlonDataset, download_and_extract, extractall
from monai.config import print_config, DtypeLike, KeysCollection
from monai.data import DataLoader, Dataset, decollate_batch, CacheNTransDataset, PersistentDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImage,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensor,
    ToTensord,
    AddChanneld,
    Resized,
    EnsureTyped,
    EnsureType,
    ResizeWithPadOrCropd,
)
from monai.transforms.transform import Transform
from monai.utils import set_determinism
from monai.utils.misc import ensure_tuple_rep

from torchvision import transforms

import torch

class ConvertToMultiChannelBasedOnDHCPClassesd(MapTransform):
    """
    Convert labels to multi channels based on 9 classes of the 
    Draw-EM 9 algorithm
    """

    def __call__(self, data):
        d = dict(data)
        # keys of the dict are: dict_keys(['image', 'label', 'image_meta_dict', 'label_meta_dict'])
        # shape of d['label'] before transformation is (217, 290, 290)
        # multi-labels: can contain 0,1,2,3,4,5,6,7,8,9 (0=background)
        
        for key in self.keys: # in our example, this is only the key 'label'
              
            result = []

            for i in range(10):
              result.append(d[key]==i)
            
            d[key] = np.stack(result, axis=0).astype(np.float32)

        return d


class HideLabelsd(MapTransform):
    """
    Hide label slices
    """

    def __init__(
            self,
            keys: KeysCollection,
            slicing_mode: str = "random",
            selection_mode: str = "random",
            proportion: float = 0.67,
            meta_key_postfix: str = "slice_dict",
            meta_key_postfix_aux: str = "slice_matrix",
            allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            slicing_mode: {``"axial"``, ``"coronal"``, ``"sagittal"``, ``"random"``}
                on which axis the 3d image is sliced
                if "random" is chosen, the axis will be chosen randomly per item
            selection_mode: {``"random"``, ``"fixed"``}
                how the slices are chosen
                if "fixed" is chosen, slices will be distributed evenly over the
                axis
            proportion: proportion of image that is hidden. should be float between
                0 and 1

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``. #not yet implemented
            ValueError: When ``proportion`` is not a float between 0 and 1

        """
        super().__init__(keys, allow_missing_keys)
        self.slicing_mode = slicing_mode
        self.selection_mode = selection_mode
        if (proportion <= 0 or proportion >= 1):
            raise ValueError("proportion must be float between 0 and 1.")
        self.proportion = proportion
        self.meta_key_postfix = meta_key_postfix
        self.meta_key_postfix_aux = meta_key_postfix_aux

        self.hide_labels_transform = HideLabels(slicing_mode, selection_mode, proportion)

    def __call__(self, data):

        d = dict(data)


        # for key, slicing_mode, selection_mode, proportion in self.key_iterator(d, self.slicing_mode, self.selection_mode, self.proportion):

        for key in self.keys:
            
            # create new keys for slice meta data and label slice matrix
            meta_key_slice_dict = f"{key}_{self.meta_key_postfix}"
            meta_key_slice_matrix = f"{key}_{self.meta_key_postfix_aux}"
            # create metadata if necessary
            if meta_key_slice_dict not in d:
                d[meta_key_slice_dict] = {}
            
            # get location, selected slices and slice_matrix from vanilla HideLabels transform
            location, selected_slices, slice_matrix = self.hide_labels_transform(d[key])

            # save down meta data and slice matrix in newly created keys/key dicts
            d[meta_key_slice_dict]["location"] = location
            d[meta_key_slice_matrix] = slice_matrix

        return d

class HideLabels(Transform):
    """

    """

    def __init__(self, slicing_mode: str = "random", selection_mode: str = "random",
                 proportion: float = 0.7) -> None:

        """
        Args:
            mode:
            selection_mode:
            proportion:
        """
        self.slicing_mode = slicing_mode
        self.selection_mode = selection_mode
        if (proportion <= 0 or proportion >= 1):
            raise ValueError("proportion must be float between 0 and 1.")
        self.proportion = proportion

    def __call__(self, img: np.ndarray) -> np.ndarray:

        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        shape = img.shape  # image has channel as first shape. we cut this out and add in the end
        result = self.get_slices_and_matrix(shape[1:])  # this will be (1, dim1, dim2, dim3)

        return result  # returns tuple of location, selected_slices and aux_label_matrix

    def get_slices_and_matrix(self, shape, slicing_mode=None):

        if slicing_mode is None:
            slicing_mode = self.slicing_mode

        if slicing_mode == "random":
            slicing_mode = np.random.choice(["axial", "sagittal", "coronal"])
            return self.get_slices_and_matrix(shape, slicing_mode=slicing_mode)

        elif slicing_mode == "sagittal":
            print("sagittal")
            loc = 0  # first dim is sagittal
            max_size = shape[0]
            lower = int(max_size * 0.05)
            upper = int(max_size * 0.95)

            number_of_slices = int(max_size * (1 - self.proportion))

            if self.selection_mode == "random":
                selected_slices = np.array(random.sample(range(lower, upper), k=number_of_slices))
            elif self.selection_mode == "fixed":
                selected_slices = np.linspace(0, max_size, number_of_slices + 2, dtype=int)[1:-1]

            # create vector which is then repeated over the other axes
            aux_vector = np.array([1 if i in selected_slices else 0 for i in range(max_size)])
            # create matrix, for tiling permutate axes
            axis_permutation = (shape[1], shape[2], 1)  # move 1 to the left
            aux_matrix = np.tile(aux_vector, axis_permutation)
            # reshape matrix to original shape
            aux_matrix_reshaped = np.transpose(aux_matrix, (2, 0, 1))

            #assert aux_matrix_reshaped.shape == shape


        elif slicing_mode == "coronal":
            print("coronal")
            loc = 1  # second dim is coronal
            max_size = shape[1]
            lower = int(max_size * 0.05)
            upper = int(max_size * 0.95)

            number_of_slices = int(max_size * (1 - self.proportion))

            if self.selection_mode == "random":
                selected_slices = np.array(random.sample(range(lower, upper), k=number_of_slices))
            elif self.selection_mode == "fixed":
                selected_slices = np.linspace(0, max_size, number_of_slices + 2, dtype=int)[1:-1]

            # create vector which is then repeated over the other axes
            aux_vector = np.array([1 if i in selected_slices else 0 for i in range(max_size)])
            # create matrix, for tiling permutate axes
            axis_permutation = (shape[2], shape[0], 1)  # move 1 to the right
            aux_matrix = np.tile(aux_vector, axis_permutation)
            # reshape matrix to original shape
            aux_matrix_reshaped = np.transpose(aux_matrix, (1, 2, 0))

            #assert aux_matrix_reshaped.shape == shape

        elif slicing_mode == "axial":
            print("axial")
            loc = 2  # third dim is axial
            max_size = shape[2]
            lower = int(max_size * 0.05)
            upper = int(max_size * 0.95)

            number_of_slices = int(max_size * (1 - self.proportion))

            if self.selection_mode == "random":  # second dimension is coronal
                selected_slices = np.array(
                    random.sample(range(lower, upper), k=number_of_slices))  # 10 slices per 3d image
            elif self.selection_mode == "fixed":
                selected_slices = np.linspace(0, max_size, number_of_slices + 2, dtype=int)[1:-1]

            # create vector which is then repeated over the other axes
            aux_vector = np.array([1 if i in selected_slices else 0 for i in range(max_size)])
            # create matrix, for tiling permutate axes
            axis_permutation = (shape[0], shape[1], 1)  # move 1 to the right
            aux_matrix = np.tile(aux_vector, axis_permutation)
            # reshape matrix to original shape --> for axial slices no transposition is necessary
            aux_matrix_reshaped = aux_matrix

        return loc, selected_slices, aux_matrix_reshaped[np.newaxis, :,:,:]  # add first dimension