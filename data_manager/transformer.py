import torch
import numpy as np

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
def image_to_tensor(image, mean=0, std=1.):  #
    """
    Transforms an image to a tensor
    Args:
        image (np.ndarray): A RGB array image
        mean: The mean of the image values
        std: The standard deviation of the image values
    Returns:
        tensor: A Pytorch tensor
    """
    image = image.astype(np.float32)
    # image = (image - mean) / std
    image[:, :, 0] = (image[:,:,0] - mean[0])/std[0]
    image[:, :, 1] = (image[:,:,1] - mean[1])/std[1]
    image[:, :, 2] = (image[:,:,2] - mean[2])/std[2]
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image)
    return tensor


def mask_to_tensor(mask, threshold):
    """
    Transforms a mask to a tensor
    Args:
        mask (np.ndarray): A greyscale mask array
        threshold: The threshold used to consider the mask present or not
    Returns:
        tensor: A Pytorch tensor
    """
    mask = mask
    mask = (mask > threshold).astype(np.float32)
    tensor = torch.from_numpy(mask).type(torch.FloatTensor)
    return tensor