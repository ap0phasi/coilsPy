import torch

from .generate_complex_sum import generate_complex_sum

def initialize_transition_tensor(num_elements, 
                                 theta, 
                                 sigma_re_lo_lim, 
                                 sigma_re_hi_lim,
                                 sigma_im,
                                 mag_re, 
                                 mag_im,
                                 restrict_dict = []):
    # Create a list to store each column
    columns = []
    num_columns = num_elements

    for col in range(num_columns):
        
        if col in restrict_dict:
            restrictions = restrict_dict[col]
        else:
            restrictions = []
                    
        column = generate_complex_sum(theta, 
                    num_elements = num_elements, 
                    center_index_re = col, 
                    center_index_im = 0, 
                    sigma_re = torch.rand(1)[0]*(sigma_re_hi_lim + sigma_re_lo_lim) + sigma_re_lo_lim, 
                    sigma_im = sigma_im, 
                    magnitude_re = mag_re, 
                    magnitude_im = mag_im,
                    restrictions = restrictions)
        columns.append(column.unsqueeze(1))  # Add a dimension to make it a column

    # Combine columns into a matrix
    transition_tensor = torch.cat(columns, dim=1)
    
    return transition_tensor


def initialize_interaction_tensor(num_elements, 
                                 theta, 
                                 sigma_re_lo_lim, 
                                 sigma_re_hi_lim,
                                 sigma_im,
                                 mag_re, 
                                 mag_im,
                                 restrict_dict = []):
    # Example usage
    num_3d_tensors = num_elements + 1  # Number of 3D tensors in the 4D tensor
    num_matrices = num_elements  # Number of 2D matrices in each 3D tensor
    num_columns = num_elements  # Number of columns in each 2D matrix

    # Create a list to store each 3D tensor
    tensors_3d = []

    for _ in range(num_3d_tensors):
        matrices = []
        for _ in range(num_matrices):
            columns = []
            for col in range(num_columns):
                
                if col in restrict_dict:
                    restrictions = restrict_dict[col]
                else:
                    restrictions = []
                    
                column = generate_complex_sum(theta, 
                                    num_elements = num_elements, 
                                    center_index_re = col, 
                                    center_index_im = 0, 
                                    sigma_re = torch.rand(1)[0]*(sigma_re_hi_lim + sigma_re_lo_lim) + sigma_re_lo_lim, 
                                    sigma_im = sigma_im, 
                                    magnitude_re = mag_re, 
                                    magnitude_im = mag_im,
                                    restrictions = restrictions)
                columns.append(column.unsqueeze(1))  # Add a dimension to make it a column
            matrix = torch.cat(columns, dim=1)
            matrices.append(matrix)  # Add a dimension to make it a 2D matrix
        tensor_3d = torch.stack(matrices, dim=2)
        tensors_3d.append(tensor_3d)

    # Combine 3D tensors into a 4D tensor
    # Stack along the fourth dimension
    interaction_tensor = torch.stack(tensors_3d, dim=3)
    
    return interaction_tensor