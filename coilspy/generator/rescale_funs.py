import torch
from torch import Tensor

def renormalize_to_unit_circle(array: Tensor) -> Tensor:
    """
    Renormalizes an array of complex numbers so that it sums to a point on the unit circle.

    Parameters:
    array (torch.Tensor): A 1D tensor of complex numbers.

    Returns:
    torch.Tensor: A renormalized 1D tensor of complex numbers.
    """
    current_sum = torch.sum(array)
    
    # Calculate the magnitude of the current sum
    magnitude = torch.abs(current_sum)

    # Calculate the scale factor
    scale_factor = 1 / magnitude

    # Renormalize the array
    renormalized_array = array * scale_factor

    return renormalized_array

def rescale_array_to_target_torch(tensor: Tensor, target_sum: float) -> Tensor:
    """
    Rescales the elements of a given tensor to collectively sum up to a target value. 
    The adjustment is only applied to non-zero elements of the tensor.

    This function computes the sum of non-zero elements in the input tensor and 
    then uniformly adjusts these non-zero elements such that their total sum matches 
    the target sum specified. Elements that are effectively zero (smaller than 1e-5) 
    are not adjusted. If all elements are zero, the tensor is returned as is.

    Parameters:
    - tensor (Tensor): A PyTorch tensor whose elements are to be rescaled.
    - target_sum (float): The desired total sum of the tensor's elements after rescaling.

    Returns:
    - Tensor: A tensor with the same shape as the input, with non-zero elements 
      rescaled to reach the target sum.

    Example:
    ```python
    # Example tensor
    tensor = torch.tensor([0.1, 0.2, 0.3, 0.4])

    # Rescale tensor to have a sum of 2.0
    rescaled_tensor = rescale_array_to_target_torch(tensor, 2.0)
    ```

    Note:
    - The function is intended for use with tensors where non-zero elements are 
      significantly larger than 1e-5.
    - If all elements of the tensor are zero (or effectively zero), no rescaling is performed.
    """
    # Create a mask for non-zero elements
    non_zero_mask = tensor > 1e-5

    # Calculate the sum of non-zero elements
    sum_non_zero = torch.sum(tensor[non_zero_mask])

    # Count the non-zero elements
    count_non_zero = torch.sum(non_zero_mask)
    
    # Avoid division by zero in case all elements are zero
    if count_non_zero == 0:
        return tensor

    # Calculate the required adjustment to reach the target sum
    adjustment = (target_sum - sum_non_zero) / count_non_zero
    
    # Add the adjustment to each non-zero element
    tensor[non_zero_mask] += adjustment

    return tensor

if __name__ == "__main__":
    # Example Usage
    
    tensor = torch.tensor([0.1, 0.2, 0.3, 0.4])
    
    rescaled_tensor = rescale_array_to_target_torch(tensor, 2.0)
