import torch
from typing import List

from .generate_kernel import generate_kernel
from .rescale_funs import rescale_array_to_target_torch

def dirichlet_to_goal(goal_value: float, num_elements: int, center_index: int, sigma: float, magnitude: float, restrictions: List[int]) -> torch.Tensor:
    """
    Generates a tensor of values sampled from a Dirichlet distribution, rescaled to meet a specified goal value.

    This function generates a kernel using a radial basis function, scales it by a magnitude, and uses
    it as alpha parameters for a Dirichlet distribution. The sampled values are then rescaled to sum up
    to a specified goal value.

    Parameters:
    - goal_value (float): The target sum for the sampled values.
    - num_elements (int): The number of elements in the output tensor.
    - center_index (int): The index at which the RBF kernel is centered.
    - sigma (float): The standard deviation (width) of the RBF kernel.
    - magnitude (float): The magnitude factor to scale the kernel.
    - restrictions (List[int]): Indices in the kernel to be set to minimal value.

    Returns:
    - torch.Tensor: A 1D tensor of values sampled from a Dirichlet distribution and rescaled.

    Example:
    ```python
    # Example usage
    dirichlet_values = dirichlet_to_goal(1.0, 5, 2, 1.0, 1.0, [0, 4])
    ```
    """
    # Generate alphas for dirichlet
    alphas = generate_kernel(length=num_elements, center_index=center_index, type="rbf", sigma=sigma, restrictions=restrictions) * magnitude
    rand_samples = torch.distributions.Dirichlet(alphas).sample()
    
    rand_samples = rescale_array_to_target_torch(rand_samples, goal_value)
    
    return rand_samples

def generate_complex_sum(goal_theta: torch.Tensor, num_elements: int, center_index_re: int, center_index_im: int, sigma_re: float, sigma_im: float, magnitude_re: float, magnitude_im: float, restrictions: List[int] = []) -> torch.Tensor:
    """
    Generates a tensor of complex numbers whose sum approximates a given complex number.

    This function generates real and imaginary parts separately using the `dirichlet_to_goal` function.
    These parts are combined to form complex numbers. The sum of these complex numbers is then rotated
    to approximate the angle specified by 'goal_theta'.

    Parameters:
    - goal_theta (torch.Tensor): The target angle for the sum of complex numbers.
    - num_elements (int): The number of complex numbers to generate.
    - center_index_re (int): The center index for the RBF kernel for real parts.
    - center_index_im (int): The center index for the RBF kernel for imaginary parts.
    - sigma_re (float): The standard deviation for the RBF kernel for real parts.
    - sigma_im (float): The standard deviation for the RBF kernel for imaginary parts.
    - magnitude_re (float): The magnitude factor for the RBF kernel for real parts.
    - magnitude_im (float): The magnitude factor for the RBF kernel for imaginary parts.
    - restrictions (List[int], optional): Indices in the kernel to be set to minimal value. Defaults to an empty list.

    Returns:
    - torch.Tensor: A tensor of complex numbers.

    Example:
    ```python
    # Example usage
    theta = torch.tensor([0.3 * torch.pi])  # Approximately 90 degrees
    complex_array = generate_complex_sum(theta, 5, 0, 0, 1e100, 1e100, 1, 1)
    ```
    """
    real_goal = torch.cos(goal_theta)
    imag_goal = torch.sin(goal_theta)
    real_parts = dirichlet_to_goal(1, num_elements, center_index_re, sigma_re, magnitude_re, restrictions)
    imag_parts = dirichlet_to_goal(0, num_elements, center_index_im, sigma_im, magnitude_im, restrictions)
    complex_numbers = torch.complex(real_parts, imag_parts)
    
    # Rotate the entire set of vectors by the goal theta
    rotated_complex_numbers = complex_numbers * (real_goal + imag_goal * 1j)
    
    rotated_complex_numbers = torch.tensor(rotated_complex_numbers, dtype=torch.complex128)

    return rotated_complex_numbers

if __name__ == "__main__":
    from viz.plot_complex_vectors import plot_complex_vectors
    # Example usage
    theta = torch.tensor([0.3 * torch.pi])  # 90 degrees, should sum to i
    complex_array = generate_complex_sum(theta, 
                                        num_elements = 5, 
                                        center_index_re = 0, 
                                        center_index_im = 0, 
                                        sigma_re = 1e100, 
                                        sigma_im = 1e100, 
                                        magnitude_re = 1, 
                                        magnitude_im = 1,
                                        restrictions = [])
    plot_complex_vectors(complex_array.numpy())
    