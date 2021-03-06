import random
import numpy as np

from Hyperparams import label_dispersion_factor
from monai.config import KeysCollection
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
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
    AddChanneld,
    Resized,
    EnsureTyped,
    EnsureType,
    ResizeWithPadOrCropd,
    ConcatItemsd,
    DeleteItemsd,
    Rand3DElasticd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    CropForegroundd,
    SpatialPadd
)
from monai.transforms.transform import Transform

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
            proportion: float = 1-label_dispersion_factor,
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
                 proportion: float = 1-label_dispersion_factor) -> None:

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


visualisation_transform = Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]), # (217, 290, 290) orig size
        AddChanneld(keys=["t1_image", "t2_image", "label"]), 
        ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
        DeleteItemsd(keys=["t1_image", "t2_image"]),
        ToTensord(keys=["image", "label"]),
    ]
)

def create_save_transform(pixdim):
    save_transform = Compose(
        [
            LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
            ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"), #(10, 217, 290, 290)
            AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
            Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
            NormalizeIntensityd(keys=["t1_image", "t2_image"], nonzero=True, channel_wise=True),
            ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
            DeleteItemsd(keys=["t1_image", "t2_image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    return save_transform

def create_train_val_transform(pixdim, roi_size, hide_labels, slicing_mode = None, selection_mode = None):

    if hide_labels:
        train_transform= Compose(
            [
            LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
            ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"), #(10, 217, 290, 290)
            AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
            Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
            HideLabelsd(keys="label", slicing_mode = slicing_mode, selection_mode = selection_mode), # define slicing_mode and selection_mode
            CropForegroundd(keys=["t1_image", "t2_image", "label", "label_slice_matrix"], source_key="t2_image", select_fn=lambda x: x>1, margin=0),
            ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
            DeleteItemsd(keys=["t1_image", "t2_image"]),
            RandSpatialCropd(
                keys=["image", "label", "label_slice_matrix"], roi_size=roi_size, random_size=False, random_center=True
            ), # [192, 192, 192]
            RandFlipd(keys=["image", "label", "label_slice_matrix"], prob=0.1, spatial_axis=0),
            RandFlipd(keys=["image", "label", "label_slice_matrix"], prob=0.1, spatial_axis=1),
            RandFlipd(keys=["image", "label", "label_slice_matrix"], prob=0.1, spatial_axis=2),
            Rand3DElasticd(
                keys=["image", "label", "label_slice_matrix"],
                mode=("bilinear", "nearest", "nearest"),
                prob=0.24,
                sigma_range=(5, 8),
                magnitude_range=(40, 80),
                translate_range=(20, 20, 20),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="reflection",
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.13),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.13,
            ),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.24),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.24),
            EnsureTyped(keys=["image", "label"]),
        ]
        )

    else:
        train_transform= Compose(
        [
            LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
            ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"), #(10, 217, 290, 290)
            AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
            Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
            CropForegroundd(keys=["t1_image", "t2_image", "label"], source_key="t2_image", select_fn=lambda x: x>1, margin=0),
            ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
            DeleteItemsd(keys=["t1_image", "t2_image"]),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=roi_size, random_size=False, random_center=True
            ), # [192, 192, 192]
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
            Rand3DElasticd(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.24,
                sigma_range=(5, 8),
                magnitude_range=(40, 80),
                translate_range=(20, 20, 20),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="reflection",
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.13),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.13,
            ),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.24),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.24),
            EnsureTyped(keys=["image", "label"]),
        ]
        )


    val_transform = Compose(
        [
            LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
            ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"),
            AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
            Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
            CropForegroundd(keys=["t1_image", "t2_image", "label"], source_key="t2_image", select_fn=lambda x: x>1, margin=0),
            ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
            DeleteItemsd(keys=["t1_image", "t2_image"]),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=roi_size, random_size=False, random_center=True
            ), 
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        
        ]
    )

    return train_transform, val_transform 

def create_test_transform(pixdim):

    test_transform= Compose(
    [
        LoadImaged(keys=["t1_image", "t2_image", "label"]), #[217, 290, 290]
        ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"), #(10, 217, 290, 290)
        AddChanneld(keys=["t1_image", "t2_image"]), #(2, 217, 290, 290)
        Spacingd(keys=["t1_image", "t2_image", "label"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
        NormalizeIntensityd(keys=["t1_image", "t2_image"], nonzero=True, channel_wise=True),
        ConcatItemsd(keys=["t1_image", "t2_image"], name="image"),
        DeleteItemsd(keys=["t1_image", "t2_image"]),
        EnsureTyped(keys=["image", "label"]),
    ])

    return test_transform

post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
