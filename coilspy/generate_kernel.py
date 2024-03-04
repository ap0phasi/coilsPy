import torch
from typing import List, Union

def generate_rbf(length: int, center_index: int, sigma: float) -> torch.Tensor:
    """
    Generates a 1D radial basis function (Gaussian) of specified width centered on a given index.

    Parameters:
    length (int): The length of the output tensor.
    center_index (int): The index at which the RBF is centered.
    sigma (float): The standard deviation (width) of the RBF.

    Returns:
    torch.Tensor: A 1D tensor representing the radial basis function.
    """

    # Create an array of indices
    x = torch.arange(0, length, dtype=torch.float32)

    # Calculate the RBF
    rbf = torch.exp(-0.5 * ((x - center_index) / sigma) ** 2)

    return rbf

def generate_kernel(length: int, center_index: int, type: str = "rbf", restrictions: List[int] = [], **kwargs) -> torch.Tensor:
    """
    Generates a kernel tensor based on the specified type and parameters. 
    Currently supports Radial Basis Function (RBF) kernel generation.

    The function allows the creation of a 1D kernel with specified characteristics, 
    such as length, center index, and type. Additional parameters for specific 
    kernel types can be passed through `kwargs`. The function also supports the 
    application of restrictions, setting specific indices in the kernel to a 
    minimal value.

    Parameters:
    - length (int): The length of the output tensor.
    - center_index (int): The index at which the kernel (e.g., RBF) is centered.
    - type (str, optional): The type of kernel to generate. Defaults to "rbf" for Radial Basis Function.
    - restrictions (list of int, optional): A list of indices in the kernel to be set to a minimal value (1e-10). 
      Defaults to an empty list, indicating no restrictions.
    - **kwargs: Additional keyword arguments for specific types of kernels. For example, 'sigma' (float) 
      is required for RBF.

    Returns:
    - torch.Tensor: A 1D tensor representing the generated kernel.

    Raises:
    - ValueError: If 'center_index' is not within the range of 'length'.
    - AssertionError: If input types do not match the expected types. 
    - KeyError: If required kwargs for a specific kernel type are missing.

    Example:
    ```python
    # Generate a Radial Basis Function kernel of length 10, centered at index 5 with a sigma of 2.0
    rbf_kernel = generate_kernel(length=10, center_index=5, type="rbf", sigma=2.0)
    ```

    Note:
    The function is currently limited to RBF kernel generation but can be extended to support other types.
    """
  
    if type == "rbf":
        kernel = generate_rbf(length, center_index, kwargs['sigma'])
        
    # Replace zero values with min_value
    kernel = torch.where(kernel == 0, torch.tensor(1e-10, dtype=kernel.dtype), kernel)
    
    # Set restrictions to 0
    kernel[restrictions] = 1e-10
    
    return kernel

if __name__ == "__main__":
    # Example Usage
    import matplotlib.pyplot as plt
    
    length = 10  # Length of the tensor
    center_index = 5  # Center the RBF at index 5
    sigma = 1.0  # Standard deviation of the RBF

    rbf_tensor = generate_kernel(length, center_index, type = "rbf", sigma=sigma)

    # Plotting the RBF
    plt.scatter(torch.arange(length), rbf_tensor)
    plt.title("Radial Basis Function (RBF)")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()