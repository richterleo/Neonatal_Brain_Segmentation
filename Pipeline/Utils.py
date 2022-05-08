import numpy as np
import torch

def get_slices_from_matrix(label_matrix, location, outputs, labels):
    """
    auxiliary function to get locations of annotated slices from label_matrix
    and slices both segmentation outputs and labels

    Args:
        label_matrix: torch.Tensor
            auxiliary matrix that captures the preprocessing/data augmentation
            of the annotated slices. shape (B, N, H, W, D) (same as image/label)

        location: torch.Tensor
            which axis to slice. shape (B, 1) (per batch)
        
        outputs: torch.Tensor
            segmentation output of model
        
        labels: torch.Tensor 
            image label

    Returns:
        sliced output and label
    """
    out_list = []
    label_list = []

    batch_size = label_matrix.shape[0]

    for i in range(batch_size):
        lab_mat = label_matrix[i][0]
        loc = location[i]
        output = outputs[i]
        label = labels[i]
        
        
        if loc== 0: #sagittal
            lab_mat = lab_mat.cpu()
            lab_mat_reversed = np.transpose(lab_mat,(1, 2, 0))
            # we only want the cols where the matrix is nonzero
            first_row = lab_mat_reversed[0,0,:].numpy()
            slices = np.nonzero(first_row)[0]
            slices = torch.from_numpy(slices)
            # slice outputs and labels
            if output.shape[0] == 4:
                out = output[:,:,slices,:,:] 
            else:
                out = output[:,slices,:,:]
            label = label[:,slices,:,:]
            
        elif loc == 1: #coronal
            lab_mat= lab_mat.cpu()
            lab_mat_reversed = np.transpose(lab_mat, (2, 0, 1)) 
            first_row = lab_mat_reversed[0,0,:].numpy()
            slices = np.nonzero(first_row)[0]
            slices = torch.from_numpy(slices)
            if output.shape[0] == 4:
                out = output[:,:,:,slices,:]
            else:
                out = output[:,:,slices,:]
            label = label[:,:,slices,:]

        elif loc == 2: #axial
            lab_mat_reversed = lab_mat.cpu()
            first_row = lab_mat_reversed[0,0,:].numpy()
            slices = np.nonzero(first_row)[0]
            slices = torch.from_numpy(slices)
            if output.shape[0] == 4:
                out = output[:,:,:,:,slices]
            else:
                out = output[:,:,:,slices]
            label = label[:,:,:,slices]
        
        out_list.append(out)
        label_list.append(label)
    
    #cannot stack tensors because they might have different sizes
    return out_list, label_list


def create_indices(data, prop_of_whole=1, val_frac=0.1, test_frac=0.1):
    """
    Function to create train/val/test split.

    Args:
        data: list of 

    """
    assert 0 < val_frac < 1, "val_frac must be larger than 0 and smaller than 1."
    assert test_frac < 1, "test_frac must be smaller than 1."

    length = int(prop_of_whole * len(data)) #use specified amount of whole dataset

    indices = np.arange(length) #list [0,...,length]
    np.random.shuffle(indices) #list is shuffled in-place

    if test_frac == 0:
        val_split = int(val_frac * length)
        val_indices = indices[:val_split]
        train_indices = indices[val_split:]
        test_indices = test_dict = None
    
    else:
        test_split = int(test_frac * length)
        val_split = int(val_frac * length) + test_split
        test_indices = indices[:test_split]
        val_indices = indices[test_split:val_split]
        train_indices = indices[val_split:]
        test_dict = [data[i] for i in test_indices]

    train_dict = [data[i] for i in train_indices] #these are just lists of file paths
    val_dict = [data[i] for i in val_indices]
    
    return train_indices, val_indices, test_indices, train_dict, val_dict, test_dict


def get_kernels_strides(patch_sizes, spacings):
    """
    function to return number of kernels and strides for DynUnet
    adapted from https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_pipeline/create_network.py

    Args: 
        patch_sizes
            list of length #dimensions, contains patch size that should be used
        spacings
            list of length #dimensions, contains voxel spacing that should be used 
    
    Returns:
        (downsample) kernel sizes and strides of DynUnet 

    """
    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, patch_sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        patch_sizes = [i / j for i, j in zip(patch_sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

def set_parameter_requires_grad(model, strategy, network_size):
    '''
    sets requires_grad attribute to false for network parameters that should be frozen
    during fine-tuning

    '''
    param_names_to_update = []
    params_to_update = []

    if network_size == 'small':
        tuning_dict = tuning_dict_small
    elif network_size == 'big':
        tuning_dict = tuning_dict_big 

    if strategy == 'no_fine_tuning': # do not train network at all
        for param in model.parameters():
            param.requires_grad = False
    
    elif strategy == 'fine_tuning':
        for param in model.parameters():
            params_to_update.append(param)
    
    else:
        for param_name, param in model.named_parameters():
            if param_name in tuning_dict[strategy]:
                param_names_to_update.append(param_name)
                params_to_update.append(param)
            else:
                param.requires_grad = False

    return param_names_to_update, params_to_update
    
    
    #if strategy is 'fine_tune', then do nothing. all params should have requires_grad
    # set to True by default


tuning_dict_small = {}
tuning_dict_big = {}

tuning_dict_small["deep"] = ['downsamples.2.conv1.conv.weight',
                            'downsamples.2.conv2.conv.weight',
                            'downsamples.2.conv3.conv.weight',
                            'downsamples.2.norm1.weight',
                            'downsamples.2.norm1.bias',
                            'downsamples.2.norm2.weight',
                            'downsamples.2.norm2.bias',
                            'downsamples.2.norm3.weight',
                            'downsamples.2.norm3.bias',
                            'bottleneck.conv1.conv.weight',
                            'bottleneck.conv2.conv.weight',
                            'bottleneck.conv3.conv.weight',
                            'bottleneck.norm1.weight',
                            'bottleneck.norm1.bias',
                            'bottleneck.norm2.weight',
                            'bottleneck.norm2.bias',
                            'bottleneck.norm3.weight',
                            'bottleneck.norm3.bias',
                            'upsamples.0.transp_conv.conv.weight',
                            'upsamples.0.conv_block.conv1.conv.weight',
                            'upsamples.0.conv_block.conv2.conv.weight',
                            'upsamples.0.conv_block.norm1.weight',
                            'upsamples.0.conv_block.norm1.bias',
                            'upsamples.0.conv_block.norm2.weight',
                            'upsamples.0.conv_block.norm2.bias',
                            'deep_supervision_heads.0.conv.conv.weight',
                            'deep_supervision_heads.0.conv.conv.bias',
                            'deep_supervision_heads.1.conv.conv.weight',
                            'deep_supervision_heads.1.conv.conv.bias',
                            'deep_supervision_heads.2.conv.conv.weight',
                            'deep_supervision_heads.2.conv.conv.bias']

tuning_dict_small["medium"] = ['downsamples.1.conv1.conv.weight',
                                    'downsamples.1.conv2.conv.weight',
                                    'downsamples.1.conv3.conv.weight',
                                    'downsamples.1.norm1.weight',
                                    'downsamples.1.norm1.bias',
                                    'downsamples.1.norm2.weight',
                                    'downsamples.1.norm2.bias',
                                    'downsamples.1.norm3.weight',
                                    'downsamples.1.norm3.bias',
                                    'downsamples.2.conv1.conv.weight',
                                    'downsamples.2.conv2.conv.weight',
                                    'downsamples.2.conv3.conv.weight',
                                    'downsamples.2.norm1.weight',
                                    'downsamples.2.norm1.bias',
                                    'downsamples.2.norm2.weight',
                                    'downsamples.2.norm2.bias',
                                    'downsamples.2.norm3.weight',
                                    'downsamples.2.norm3.bias',
                                    'bottleneck.conv1.conv.weight',
                                    'bottleneck.conv2.conv.weight',
                                    'bottleneck.conv3.conv.weight',
                                    'bottleneck.norm1.weight',
                                    'bottleneck.norm1.bias',
                                    'bottleneck.norm2.weight',
                                    'bottleneck.norm2.bias',
                                    'bottleneck.norm3.weight',
                                    'bottleneck.norm3.bias',
                                    'upsamples.0.transp_conv.conv.weight',
                                    'upsamples.0.conv_block.conv1.conv.weight',
                                    'upsamples.0.conv_block.conv2.conv.weight',
                                    'upsamples.0.conv_block.norm1.weight',
                                    'upsamples.0.conv_block.norm1.bias',
                                    'upsamples.0.conv_block.norm2.weight',
                                    'upsamples.0.conv_block.norm2.bias',
                                    'upsamples.1.transp_conv.conv.weight',
                                    'upsamples.1.conv_block.conv1.conv.weight',
                                    'upsamples.1.conv_block.conv2.conv.weight',
                                    'upsamples.1.conv_block.norm1.weight',
                                    'upsamples.1.conv_block.norm1.bias',
                                    'upsamples.1.conv_block.norm2.weight',
                                    'upsamples.1.conv_block.norm2.bias',
                                    'deep_supervision_heads.0.conv.conv.weight',
                                    'deep_supervision_heads.0.conv.conv.bias',
                                    'deep_supervision_heads.1.conv.conv.weight',
                                    'deep_supervision_heads.1.conv.conv.bias',
                                    'deep_supervision_heads.2.conv.conv.weight',
                                    'deep_supervision_heads.2.conv.conv.bias']

tuning_dict_small['shallow'] = ['downsamples.0.conv1.conv.weight',
                                    'downsamples.0.conv2.conv.weight',
                                    'downsamples.0.conv3.conv.weight',
                                    'downsamples.0.norm1.weight',
                                    'downsamples.0.norm1.bias',
                                    'downsamples.0.norm2.weight',
                                    'downsamples.0.norm2.bias',
                                    'downsamples.0.norm3.weight',
                                    'downsamples.0.norm3.bias',
                                    'downsamples.1.conv1.conv.weight',
                                    'downsamples.1.conv2.conv.weight',
                                    'downsamples.1.conv3.conv.weight',
                                    'downsamples.1.norm1.weight',
                                    'downsamples.1.norm1.bias',
                                    'downsamples.1.norm2.weight',
                                    'downsamples.1.norm2.bias',
                                    'downsamples.1.norm3.weight',
                                    'downsamples.1.norm3.bias',
                                    'downsamples.2.conv1.conv.weight',
                                    'downsamples.2.conv2.conv.weight',
                                    'downsamples.2.conv3.conv.weight',
                                    'downsamples.2.norm1.weight',
                                    'downsamples.2.norm1.bias',
                                    'downsamples.2.norm2.weight',
                                    'downsamples.2.norm2.bias',
                                    'downsamples.2.norm3.weight',
                                    'downsamples.2.norm3.bias',
                                    'bottleneck.conv1.conv.weight',
                                    'bottleneck.conv2.conv.weight',
                                    'bottleneck.conv3.conv.weight',
                                    'bottleneck.norm1.weight',
                                    'bottleneck.norm1.bias',
                                    'bottleneck.norm2.weight',
                                    'bottleneck.norm2.bias',
                                    'bottleneck.norm3.weight',
                                    'bottleneck.norm3.bias',
                                    'upsamples.0.transp_conv.conv.weight',
                                    'upsamples.0.conv_block.conv1.conv.weight',
                                    'upsamples.0.conv_block.conv2.conv.weight',
                                    'upsamples.0.conv_block.norm1.weight',
                                    'upsamples.0.conv_block.norm1.bias',
                                    'upsamples.0.conv_block.norm2.weight',
                                    'upsamples.0.conv_block.norm2.bias',
                                    'upsamples.1.transp_conv.conv.weight',
                                    'upsamples.1.conv_block.conv1.conv.weight',
                                    'upsamples.1.conv_block.conv2.conv.weight',
                                    'upsamples.1.conv_block.norm1.weight',
                                    'upsamples.1.conv_block.norm1.bias',
                                    'upsamples.1.conv_block.norm2.weight',
                                    'upsamples.1.conv_block.norm2.bias',
                                    'upsamples.2.transp_conv.conv.weight',
                                    'upsamples.2.conv_block.conv1.conv.weight',
                                    'upsamples.2.conv_block.conv2.conv.weight',
                                    'upsamples.2.conv_block.norm1.weight',
                                    'upsamples.2.conv_block.norm1.bias',
                                    'upsamples.2.conv_block.norm2.weight',
                                    'upsamples.2.conv_block.norm2.bias',
                                    'deep_supervision_heads.0.conv.conv.weight',
                                    'deep_supervision_heads.0.conv.conv.bias',
                                    'deep_supervision_heads.1.conv.conv.weight',
                                    'deep_supervision_heads.1.conv.conv.bias',
                                    'deep_supervision_heads.2.conv.conv.weight',
                                    'deep_supervision_heads.2.conv.conv.bias']


tuning_dict_big['deep'] = ['downsamples.3.conv1.conv.weight',
                            'downsamples.3.conv2.conv.weight',
                            'downsamples.3.conv3.conv.weight',
                            'downsamples.3.norm1.weight',
                            'downsamples.3.norm1.bias',
                            'downsamples.3.norm2.weight',
                            'downsamples.3.norm2.bias',
                            'downsamples.3.norm3.weight',
                            'downsamples.3.norm3.bias',
                            'bottleneck.conv1.conv.weight',
                            'bottleneck.conv2.conv.weight',
                            'bottleneck.conv3.conv.weight',
                            'bottleneck.norm1.weight',
                            'bottleneck.norm1.bias',
                            'bottleneck.norm2.weight',
                            'bottleneck.norm2.bias',
                            'bottleneck.norm3.weight',
                            'bottleneck.norm3.bias',
                            'upsamples.0.transp_conv.conv.weight',
                            'upsamples.0.conv_block.conv1.conv.weight',
                            'upsamples.0.conv_block.conv2.conv.weight',
                            'upsamples.0.conv_block.norm1.weight',
                            'upsamples.0.conv_block.norm1.bias',
                            'upsamples.0.conv_block.norm2.weight',
                            'upsamples.0.conv_block.norm2.bias',
                            'deep_supervision_heads.0.conv.conv.weight',
                            'deep_supervision_heads.0.conv.conv.bias',
                            'deep_supervision_heads.1.conv.conv.weight',
                            'deep_supervision_heads.1.conv.conv.bias',
                            'deep_supervision_heads.2.conv.conv.weight',
                            'deep_supervision_heads.2.conv.conv.bias',
                            'deep_supervision_heads.3.conv.conv.weight',
                            'deep_supervision_heads.3.conv.conv.bias']

tuning_dict_big['medium'] = ['downsamples.2.conv1.conv.weight',
                            'downsamples.2.conv2.conv.weight',
                            'downsamples.2.conv3.conv.weight',
                            'downsamples.2.norm1.weight',
                            'downsamples.2.norm1.bias',
                            'downsamples.2.norm2.weight',
                            'downsamples.2.norm2.bias',
                            'downsamples.2.norm3.weight',
                            'downsamples.2.norm3.bias',
                            'downsamples.3.conv1.conv.weight',
                            'downsamples.3.conv2.conv.weight',
                            'downsamples.3.conv3.conv.weight',
                            'downsamples.3.norm1.weight',
                            'downsamples.3.norm1.bias',
                            'downsamples.3.norm2.weight',
                            'downsamples.3.norm2.bias',
                            'downsamples.3.norm3.weight',
                            'downsamples.3.norm3.bias',
                            'bottleneck.conv1.conv.weight',
                            'bottleneck.conv2.conv.weight',
                            'bottleneck.conv3.conv.weight',
                            'bottleneck.norm1.weight',
                            'bottleneck.norm1.bias',
                            'bottleneck.norm2.weight',
                            'bottleneck.norm2.bias',
                            'bottleneck.norm3.weight',
                            'bottleneck.norm3.bias',
                            'upsamples.0.transp_conv.conv.weight',
                            'upsamples.0.conv_block.conv1.conv.weight',
                            'upsamples.0.conv_block.conv2.conv.weight',
                            'upsamples.0.conv_block.norm1.weight',
                            'upsamples.0.conv_block.norm1.bias',
                            'upsamples.0.conv_block.norm2.weight',
                            'upsamples.0.conv_block.norm2.bias',
                            'upsamples.1.transp_conv.conv.weight',
                            'upsamples.1.conv_block.conv1.conv.weight',
                            'upsamples.1.conv_block.conv2.conv.weight',
                            'upsamples.1.conv_block.norm1.weight',
                            'upsamples.1.conv_block.norm1.bias',
                            'upsamples.1.conv_block.norm2.weight',
                            'upsamples.1.conv_block.norm2.bias',
                            'deep_supervision_heads.0.conv.conv.weight',
                            'deep_supervision_heads.0.conv.conv.bias',
                            'deep_supervision_heads.1.conv.conv.weight',
                            'deep_supervision_heads.1.conv.conv.bias',
                            'deep_supervision_heads.2.conv.conv.weight',
                            'deep_supervision_heads.2.conv.conv.bias',
                            'deep_supervision_heads.3.conv.conv.weight',
                            'deep_supervision_heads.3.conv.conv.bias']

tuning_dict_big["shallow"] = ['downsamples.0.conv1.conv.weight',
                            'downsamples.0.conv2.conv.weight',
                            'downsamples.0.conv3.conv.weight',
                            'downsamples.0.norm1.weight',
                            'downsamples.0.norm1.bias',
                            'downsamples.0.norm2.weight',
                            'downsamples.0.norm2.bias',
                            'downsamples.0.norm3.weight',
                            'downsamples.0.norm3.bias',
                            'downsamples.1.conv1.conv.weight',
                            'downsamples.1.conv2.conv.weight',
                            'downsamples.1.conv3.conv.weight',
                            'downsamples.1.norm1.weight',
                            'downsamples.1.norm1.bias',
                            'downsamples.1.norm2.weight',
                            'downsamples.1.norm2.bias',
                            'downsamples.1.norm3.weight',
                            'downsamples.1.norm3.bias',
                            'downsamples.2.conv1.conv.weight',
                            'downsamples.2.conv2.conv.weight',
                            'downsamples.2.conv3.conv.weight',
                            'downsamples.2.norm1.weight',
                            'downsamples.2.norm1.bias',
                            'downsamples.2.norm2.weight',
                            'downsamples.2.norm2.bias',
                            'downsamples.2.norm3.weight',
                            'downsamples.2.norm3.bias',
                            'downsamples.3.conv1.conv.weight',
                            'downsamples.3.conv2.conv.weight',
                            'downsamples.3.conv3.conv.weight',
                            'downsamples.3.norm1.weight',
                            'downsamples.3.norm1.bias',
                            'downsamples.3.norm2.weight',
                            'downsamples.3.norm2.bias',
                            'downsamples.3.norm3.weight',
                            'downsamples.3.norm3.bias',
                            'bottleneck.conv1.conv.weight',
                            'bottleneck.conv2.conv.weight',
                            'bottleneck.conv3.conv.weight',
                            'bottleneck.norm1.weight',
                            'bottleneck.norm1.bias',
                            'bottleneck.norm2.weight',
                            'bottleneck.norm2.bias',
                            'bottleneck.norm3.weight',
                            'bottleneck.norm3.bias',
                            'upsamples.0.transp_conv.conv.weight',
                            'upsamples.0.conv_block.conv1.conv.weight',
                            'upsamples.0.conv_block.conv2.conv.weight',
                            'upsamples.0.conv_block.norm1.weight',
                            'upsamples.0.conv_block.norm1.bias',
                            'upsamples.0.conv_block.norm2.weight',
                            'upsamples.0.conv_block.norm2.bias',
                            'upsamples.1.transp_conv.conv.weight',
                            'upsamples.1.conv_block.conv1.conv.weight',
                            'upsamples.1.conv_block.conv2.conv.weight',
                            'upsamples.1.conv_block.norm1.weight',
                            'upsamples.1.conv_block.norm1.bias',
                            'upsamples.1.conv_block.norm2.weight',
                            'upsamples.1.conv_block.norm2.bias',
                            'upsamples.2.transp_conv.conv.weight',
                            'upsamples.2.conv_block.conv1.conv.weight',
                            'upsamples.2.conv_block.conv2.conv.weight',
                            'upsamples.2.conv_block.norm1.weight',
                            'upsamples.2.conv_block.norm1.bias',
                            'upsamples.2.conv_block.norm2.weight',
                            'upsamples.2.conv_block.norm2.bias',
                            'upsamples.3.transp_conv.conv.weight',
                            'upsamples.3.conv_block.conv1.conv.weight',
                            'upsamples.3.conv_block.conv2.conv.weight',
                            'upsamples.3.conv_block.norm1.weight',
                            'upsamples.3.conv_block.norm1.bias',
                            'upsamples.3.conv_block.norm2.weight',
                            'upsamples.3.conv_block.norm2.bias',
                            'deep_supervision_heads.0.conv.conv.weight',
                            'deep_supervision_heads.0.conv.conv.bias',
                            'deep_supervision_heads.1.conv.conv.weight',
                            'deep_supervision_heads.1.conv.conv.bias',
                            'deep_supervision_heads.2.conv.conv.weight',
                            'deep_supervision_heads.2.conv.conv.bias',
                            'deep_supervision_heads.3.conv.conv.weight',
                            'deep_supervision_heads.3.conv.conv.bias']


# Additionally trained when fine-tuning the whole small network: 
# 'input_block.conv1.conv.weight'
# 'input_block.conv2.conv.weight'
# 'input_block.conv3.conv.weight'
# 'input_block.norm1.weight'
# 'input_block.norm1.bias'
# 'input_block.norm2.weight'
# 'input_block.norm2.bias'
# 'input_block.norm3.weight'
# 'input_block.norm3.bias'
# 'upsamples.4.transp_conv.conv.weight'
# 'upsamples.4.conv_block.conv1.conv.weight'
# 'upsamples.4.conv_block.conv2.conv.weight'
# 'upsamples.4.conv_block.norm1.weight'
# 'upsamples.4.conv_block.norm1.bias'
# 'upsamples.4.conv_block.norm2.weight'
# 'upsamples.4.conv_block.norm2.bias'
# 'output_block.conv.conv.weight'
# 'output_block.conv.conv.bias'


# Additionally trained when fine-tuning the whole small network: 
# input_block.conv1.conv.weight
# input_block.conv2.conv.weight
# input_block.conv3.conv.weight
# input_block.norm1.weight
# input_block.norm1.bias
# input_block.norm2.weight
# input_block.norm2.bias
# input_block.norm3.weight
# input_block.norm3.bias
# upsamples.3.transp_conv.conv.weight
# upsamples.3.conv_block.conv1.conv.weight
# upsamples.3.conv_block.conv2.conv.weight
# upsamples.3.conv_block.norm1.weight
# upsamples.3.conv_block.norm1.bias
# upsamples.3.conv_block.norm2.weight
# upsamples.3.conv_block.norm2.bias
# output_block.conv.conv.weight
# output_block.conv.conv.bias


# aux functions
def get_slices_from_matrix(label_matrix, location, outputs, labels):
    """
    auxiliary function to get locations of annotated slices from label_matrix
    and slices both segmentation outputs and labels

    Args:
        label_matrix: torch.Tensor
            auxiliary matrix that captures the preprocessing/data augmentation
            of the annotated slices. shape (B, N, H, W, D) (same as image/label)

        location: torch.Tensor
            which axis to slice. shape (B, 1) (per batch)
        
        outputs: torch.Tensor
            segmentation output of model
        
        labels: torch.Tensor 
            image label

    Returns:
        sliced output and label
    """
    out_list = []
    label_list = []

    batch_size = label_matrix.shape[0]

    for i in range(batch_size):
        lab_mat = label_matrix[i][0]
        loc = location[i]
        output = outputs[i]
        label = labels[i]
        
        
        if loc== 0: #sagittal
            lab_mat = lab_mat.cpu()
            lab_mat_reversed = np.transpose(lab_mat,(1, 2, 0))
            # we only want the cols where the matrix is nonzero
            first_row = lab_mat_reversed[0,0,:].numpy()
            slices = np.nonzero(first_row)[0]
            slices = torch.from_numpy(slices)
            # slice outputs and labels
            if output.shape[0] == 4:
                out = output[:,:,slices,:,:] 
            else:
                out = output[:,slices,:,:]
            label = label[:,slices,:,:]
            
        elif loc == 1: #coronal
            lab_mat= lab_mat.cpu()
            #dims after reversal should be [12,10,11]. we want to look over last dim
            lab_mat_reversed = np.transpose(lab_mat, (2, 0, 1)) 
            first_row = lab_mat_reversed[0,0,:].numpy()
            slices = np.nonzero(first_row)[0]
            slices = torch.from_numpy(slices)
            if output.shape[0] == 4:
                out = output[:,:,:,slices,:]
            else:
                out = output[:,:,slices,:]
            label = label[:,:,slices,:]

        elif loc == 2: #axial
            lab_mat_reversed = lab_mat.cpu()
            first_row = lab_mat_reversed[0,0,:].numpy()
            slices = np.nonzero(first_row)[0]
            slices = torch.from_numpy(slices)
            if output.shape[0] == 4:
                out = output[:,:,:,:,slices]
            else:
                out = output[:,:,:,slices]
            label = label[:,:,:,slices]
        
        out_list.append(out)
        label_list.append(label)
    
    #cannot stack tensors because they might have different sizes
    return out_list, label_list


def create_indices(data_dict, prop_of_whole=1, val_frac=0.1, test_frac=0.1):
    """
    function to create train/val/test split

    """
    assert 0 < val_frac < 1, "val_frac must be larger than 0 and smaller than 1."
    assert test_frac < 1, "test_frac must be smaller than 1."

    length = int(prop_of_whole * len(data_dict)) #use specified amount of whole dataset

    indices = np.arange(length) #list [0,...,length]
    np.random.shuffle(indices) #list is shuffled in-place

    if test_frac == 0:
        val_split = int(val_frac * length)
        val_indices = indices[:val_split]
        train_indices = indices[val_split:]
        test_indices = test_dict = None
    
    else:
        test_split = int(test_frac * length)
        val_split = int(val_frac * length) + test_split
        test_indices = indices[:test_split]
        val_indices = indices[test_split:val_split]
        train_indices = indices[val_split:]
        test_dict = [data_dict[i] for i in test_indices]

    train_dict = [data_dict[i] for i in train_indices] #these are just lists of file paths
    val_dict = [data_dict[i] for i in val_indices]
    
    return train_indices, val_indices, test_indices, train_dict, val_dict, test_dict


def get_kernels_strides(patch_sizes, spacings):
    """
    function to return number of kernels and strides for DynUnet
    adapted from https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_pipeline/create_network.py

    Args: 
        patch_sizes
            list of length #dimensions, contains patch size that should be used
        spacings
            list of length #dimensions, contains voxel spacing that should be used 
    
    Returns:
        (downsample) kernel sizes and strides of DynUnet 

    """
    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, patch_sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        patch_sizes = [i / j for i, j in zip(patch_sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

def set_parameter_requires_grad(model, strategy, network_size):
    '''
    sets requires_grad attribute to false for network parameters that should be frozen
    during fine-tuning

    '''
    param_names_to_update = []
    params_to_update = []

    if network_size == 'small':
        tuning_dict = tuning_dict_small
    elif network_size == 'big':
        tuning_dict = tuning_dict_big 

    if strategy == 'no_fine_tuning': # do not train network at all
        for param in model.parameters():
            param.requires_grad = False
    
    elif strategy == 'fine_tuning':
        for param in model.parameters():
            params_to_update.append(param)
    
    else:
        for param_name, param in model.named_parameters():
            if param_name in tuning_dict[strategy]:
                param_names_to_update.append(param_name)
                params_to_update.append(param)
            else:
                param.requires_grad = False

    return param_names_to_update, params_to_update
    
    
    #if strategy is 'fine_tune', then do nothing. all params should have requires_grad
    # set to True by default

if __name__ == "__main__":

    # only two important cases:
    patch_sizes_high_res = [128, 128, 128]
    spacings_high_res = [0.5, 0.5, 0.5]

    high_res_kernels, high_res_strides = get_kernels_strides(patch_sizes_high_res, spacings_high_res)

    patch_sizes_low_res = [96, 96, 96]
    spacings_low_res = [0.6, 0.6, 0.6]

    low_res_kernels, low_res_strides = get_kernels_strides(patch_sizes_low_res, spacings_low_res)

    print(f"If high memory, kernels: {high_res_kernels}, spacings: {high_res_strides}")
    print(f"If low memory, kernels: {low_res_kernels}, spacings: {low_res_strides}")
