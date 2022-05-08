# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Functionalities in this module are based on monai.networks.blocks.dynunet.DynUNetSkipLayer, 
# monai.networks.nets.DynUnet and monai.inferers.sliding_window_inference
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.data import DataLoader, Dataset
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.inferers.utils import _get_scan_interval
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock
from monai.networks.layers.factories import Act
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    AddChanneld,
    Resized,
    EnsureTyped,
    ResizeWithPadOrCropd)
from Utils import get_kernels_strides
from torch.nn.functional import interpolate
from Transforms import ConvertToMultiChannelBasedOnDHCPClassesd
from typing import  Any, Callable, List, Optional, Sequence, Tuple, Union


class AgeDynUNetSkipLayer(nn.Module):
    """
    Adapted from monai.networks.blocks.dynunet.DynUNetSkipLayer for added age prediction. 

    Defines a layer in the UNet topology which combines the downsample and upsample pathways with the skip connection.
    The member `next_layer` may refer to instances of this class or the final bottleneck layer at the bottom the UNet
    structure. The purpose of using a recursive class like this is to get around the Torchscript restrictions on
    looping over lists of layers and accumulating lists of output tensors which must be indexed. The `heads` list is
    shared amongst all the instances of this class and is used to store the output from the supervision heads during
    forward passes of the network.
    """

    heads: List[torch.Tensor]

    def __init__(self, index, heads, downsample, upsample, super_head, next_layer, is_bottom):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample
        self.next_layer = next_layer
        self.super_head = super_head
        self.heads = heads
        self.index = index
        self.is_bottom = is_bottom

    def forward(self, x, age_pred = None):

        downout = self.downsample(x)

        # if we're in the bottom layer, we want to forward the value 
        # to use for age prediction
        # after bottom layer, nextout becomes a tuple (output, age_pred)
        if self.is_bottom: 
            nextout = self.next_layer(downout)
        else:
            nextout = self.next_layer(downout, age_pred)

        if isinstance(nextout, tuple):
            upout = self.upsample(nextout[0], downout)
            self.heads[self.index] = self.super_head(upout)
            upout = (upout, nextout[1])
        else:
            upout = self.upsample(nextout, downout)
            self.heads[self.index] = self.super_head(upout)

        if self.is_bottom:
            return upout, nextout
        
        else:
            return upout


class AgeDynUNet(nn.Module):
    """
    Reimplementation of a dynamic UNet (DynUNet) with added age predction. Adapted from monai.networks.nets.DynUnet, 
    which is based on
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    This model is more flexible compared with ``monai.networks.nets.UNet`` in three
    places:

        - Residual connection is supported in conv blocks.
        - Anisotropic kernel sizes and strides can be used in each layers.
        - Deep supervision heads can be added.

    The model supports 2D or 3D inputs and is consisted with four kinds of blocks:
    one input block, `n` downsample blocks, one bottleneck and `n+1` upsample blocks. Where, `n>0`.
    The first and last kernel and stride values of the input sequences are used for input block and
    bottleneck respectively, and the rest value(s) are used for downsample and upsample blocks.
    Therefore, pleasure ensure that the length of input sequences (``kernel_size`` and ``strides``)
    is no less than 3 in order to have at least one downsample and upsample blocks.

    To meet the requirements of the structure, the input size for each spatial dimension should be divisible
    by `2 * the product of all strides in the corresponding dimension`. The output size for each spatial dimension
    equals to the input size of the correponding dimension divided by the stride in strides[0].
    For example, if `strides=((1, 2, 4), 2, 1, 1)`, the minimal spatial size of the input is `(8, 16, 32)`, and
    the spatial size of the output is `(8, 8, 8)`.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        strides: convolution strides for each blocks.
        upsample_kernel_size: convolution kernel size for transposed convolution layers. The values should
            equal to strides[1:].
        norm_name: feature normalization type and arguments. Defaults to ``INSTANCE``.
        deep_supervision: whether to add deep supervision head before output. Defaults to ``False``.
            If ``True``, in training mode, the forward function will output not only the last feature
            map, but also the previous feature maps that come from the intermediate up sample layers.
            In order to unify the return type (the restriction of TorchScript), all intermediate
            feature maps are interpolated into the same size as the last feature map and stacked together
            (with a new dimension in the first axis)into one single tensor.
            For instance, if there are three feature maps with shapes: (1, 2, 32, 24), (1, 2, 16, 12) and
            (1, 2, 8, 6). The last two will be interpolated into (1, 2, 32, 24), and the stacked tensor
            will has the shape (1, 3, 2, 8, 6).
            When calculating the loss, you can use torch.unbind to get all feature maps can compute the loss
            one by one with the ground truth, then do a weighted average for all losses to achieve the final loss.
            (To be added: a corresponding tutorial link)

        deep_supr_num: number of feature maps that will output during deep supervision head. The
            value should be larger than 0 and less than the number of up sample layers.
            Defaults to 1.
        res_block: whether to use residual connection based convolution blocks during the network.
            Defaults to ``False``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
    ):
        super(AgeDynUNet, self).__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.filters = [min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))]
        self.input_block = self.get_input_block()
        self.bottleneck = self.get_bottleneck()
        self.downsamples = self.get_downsamples()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = self.get_deep_supervision_heads()
        self.deep_supr_num = deep_supr_num
        self.apply(self.initialize_weights)
        self.check_kernel_stride()
        self.check_deep_supr_num()

        # initialize the typed list of supervision head outputs so that Torchscript can recognize what's going on
        self.heads: List[torch.Tensor] = [torch.rand(1)] * (len(self.deep_supervision_heads) + 1)

        def create_skips(index, downsamples, upsamples, superheads, bottleneck):
            """
            Construct the UNet topology as a sequence of skip layers terminating with the bottleneck layer. This is
            done recursively from the top down since a recursive nn.Module subclass is being used to be compatible
            with Torchscript. Initially the length of `downsamples` will be one more than that of `superheads`
            since the `input_block` is passed to this function as the first item in `downsamples`, however this
            shouldn't be associated with a supervision head.
            """


            if len(downsamples) != len(upsamples):
                raise AssertionError(f"{len(downsamples)} != {len(upsamples)}")
            if (len(downsamples) - len(superheads)) not in (1, 0):
                raise AssertionError(f"{len(downsamples)}-(0,1) != {len(superheads)}")

            if len(downsamples) == 0:  # bottom of the network, pass the bottleneck block
                return bottleneck
                
            if index == 0:  # don't associate a supervision head with self.input_block
                current_head, rest_heads = nn.Identity(), superheads
            elif not self.deep_supervision:  # bypass supervision heads by passing nn.Identity in place of a real one
                current_head, rest_heads = nn.Identity(), superheads[1:]
            else:
                current_head, rest_heads = superheads[0], superheads[1:]

            # create the next layer down, this will stop at the bottleneck layer
            next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], rest_heads, bottleneck)

            if next_layer == bottleneck:
                is_bottom = True
            else:
                is_bottom = False

            return AgeDynUNetSkipLayer(index, self.heads, downsamples[0], upsamples[0], current_head, next_layer, is_bottom)

        self.skip_layers = create_skips(
            0,
            [self.input_block] + list(self.downsamples),
            self.upsamples[::-1],
            self.deep_supervision_heads,
            self.bottleneck,
        )

        # create extra parts of network for age prediction; use output of bottom layer
        self.age_conv1 = Convolution(
            self.spatial_dims,
            self.filters[-1],
            512,
            kernel_size=(3, 3, 3),
            padding=0,
            act=Act.PRELU,
            norm=None,
            bias=True,
        )

        self.age_conv2 = Convolution(
            self.spatial_dims,
            512,
            1024,
            kernel_size=(3, 3, 3),
            padding=0,
            act=Act.PRELU,
            norm=None,
            bias=True,
        )

        self.lin_age_block_large = nn.Linear(8192, 1)
        self.lin_age_block_small = nn.Linear(4096, 1)
        self.age_act = nn.PReLU()


    def check_kernel_stride(self):
        kernels, strides = self.kernel_size, self.strides
        error_msg = "length of kernel_size and strides should be the same, and no less than 3."
        if not (len(kernels) == len(strides) and len(kernels) >= 3):
            raise AssertionError(error_msg)

        for idx, k_i in enumerate(kernels):
            kernel, stride = k_i, strides[idx]
            if not isinstance(kernel, int):
                error_msg = "length of kernel_size in block {} should be the same as spatial_dims.".format(idx)
                if len(kernel) != self.spatial_dims:
                    raise AssertionError(error_msg)
            if not isinstance(stride, int):
                error_msg = "length of stride in block {} should be the same as spatial_dims.".format(idx)
                if len(stride) != self.spatial_dims:
                    raise AssertionError(error_msg)

    def check_deep_supr_num(self):
        deep_supr_num, strides = self.deep_supr_num, self.strides
        num_up_layers = len(strides) - 1
        if deep_supr_num >= num_up_layers:
            raise AssertionError("deep_supr_num should be less than the number of up sample layers.")
        if deep_supr_num < 1:
            raise AssertionError("deep_supr_num should be larger than 0.")

    def forward(self, x):
        out, age_x = self.skip_layers(x)
        out = self.output_block(out)
        

        # age prediction; send input through another conv layer
        if age_x.shape[-1] > 4:
            age_x = self.age_conv1(age_x) #shape [B, 512, 2, 2, 2]
            age_x = self.age_conv2(age_x)
            # flatten
            age_x_flattened = age_x.view(age_x.size(0), -1) #shape [B, 8192]
            # send through linear layer
            age_x_lin = self.lin_age_block_large(age_x_flattened) #shape [B, 1]
            age_pred = self.age_act(age_x_lin)
        else:
            age_x = self.age_conv1(age_x)
            # flatten
            age_x_flattened = age_x.view(age_x.size(0), -1) #shape [B, 4096]
            # send through linear layer
            age_x_lin = self.lin_age_block_small(age_x_flattened) #shape [B, 1]
            age_pred = self.age_act(age_x_lin)


        if self.training and self.deep_supervision:
            out_all = [out]
            feature_maps = self.heads[1 : self.deep_supr_num + 1]
            for feature_map in feature_maps:
                out_all.append(interpolate(feature_map, out.shape[2:]))
            return torch.stack(out_all, dim=1), age_pred
        
        return out, age_pred


    def get_input_block(self):
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
        )

    def get_bottleneck(self):
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
        )

    def get_output_block(self, idx: int):
        return UnetOutBlock(
            self.spatial_dims,
            self.filters[idx],
            self.out_channels,
        )

    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size)

    def get_module_list(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: nn.Module,
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
    ):
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "upsample_kernel_size": up_kernel,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(i + 1) for i in range(len(self.upsamples) - 1)])

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def sliding_window_inference_age(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Sliding window inference on `inputs` with `predictor` with additional age prediction.
    Source code adapted from monai.inferers.sliding_window_inference.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/nn.functional.html#pad
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # record additional age predictions
    age_prediction = []

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        seg_prob = predictor(window_data, *args, **kwargs) # batched patch segmentation
        
        # seg_prob is a tuple of (segmentation, age_prediction)
        age_prediction.append(seg_prob[1][0][0])
        seg_prob = seg_prob[0].to(device) 

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            count_map[original_idx] += importance_map

    # account for any overlapping sections
    output_image = output_image / count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))

    return output_image[final_slicing], torch.mean(torch.Tensor(age_prediction))



if __name__ == "__main__":
    device = torch.device("cuda:0")

    high_ram = False
    if high_ram:
        patch_sizes = [128, 128, 128]
        spacings = [0.5, 0.5, 0.5]
    else:
        patch_sizes = [96, 96, 96]
        spacings = [0.6, 0.6, 0.6]

    kernels, strides = get_kernels_strides(patch_sizes, spacings)

    agedynunet = AgeDynUNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=10,
                    kernel_size=kernels,
                    strides=strides,
                    upsample_kernel_size=strides[1:],
                    deep_supervision=True,
                    deep_supr_num = 3,
                    res_block=True
                ).to(device)

    root_dir = os.path.join(os.getcwd(), 'dHCP_Training')

    data_dir = os.path.join(root_dir, 'backup_dHCP') 
    t1_dir = os.path.join(data_dir, 'T1w')
    label_dir = os.path.join(data_dir, 'labels')

    #Create list of files
    t1_list = sorted([os.path.join(t1_dir, file) for file in os.listdir(t1_dir)])
    label_list = sorted([os.path.join(label_dir, file) for file in os.listdir(label_dir)])

    data_dict = [{"t1_image": t1_image, "label": label} for t1_image, label in zip(t1_list, label_list)]
    data_dict = data_dict[:1]
    print(f"we have {len(data_dict)} train images")

    transform = Compose(
        [
            LoadImaged(keys=["t1_image", "label"]), #[217, 290, 290]
            ConvertToMultiChannelBasedOnDHCPClassesd(keys="label"),
            AddChanneld(keys="t1_image"), #(2, 217, 290, 290)
            ResizeWithPadOrCropd(keys=["t1_image", "label"], spatial_size=[290,290,290], mode="edge"), #(1, 290, 290, 290)
            Resized(keys=["t1_image", "label"], spatial_size=patch_sizes), # (1, 128, 128, 64)
            NormalizeIntensityd(keys="t1_image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["t1_image", "label"]),
        
        ]
    )

    ds = Dataset(data_dict, transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    print(f"The ds has {len(ds)} elements")

    for batch_data in loader:
        inputs, labels = (
        batch_data["t1_image"].to(device),
        batch_data["label"].to(device)
        )
        outputs, age_pred = agedynunet(inputs)
        print(f"outputs have shape: {outputs.shape}")
        print(f"age pred shape hier unten: {age_pred.shape}")