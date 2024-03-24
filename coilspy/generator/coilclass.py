import torch
from .initialization import initialize_interaction_tensor, initialize_transition_tensor
from .generate_complex_sum import generate_complex_sum
from .selector import select_transition_tensor
from .rescale_funs import renormalize_to_unit_circle

class ComplexCoil:
    def __init__(self,
                 num_elements,
                 angles_dict, 
                 state_center_re,
                 state_center_im,
                 state_sigma_re,
                 state_sigma_im,
                 state_magnitude_re,
                 state_magnitude_im,
                 state_restrictions,
                 trans_sigma_re_lo_lim,
                 trans_sigma_re_hi_lim,
                 trans_sigma_im,
                 trans_magnitude_re,
                 trans_magnitude_im,
                 trans_restrictions,
                 inter_sigma_re_lo_lim,
                 inter_sigma_re_hi_lim,
                 inter_sigma_im,
                 inter_magnitude_re,
                 inter_magnitude_im,
                 inter_restrictions,
                 ) -> None:
        
        self.num_elements = num_elements
        self.thetas_dict = {key : torch.tensor([value * torch.pi]) for key, value in angles_dict.items()}
        
        self.state_tensor = generate_complex_sum(
                                     self.thetas_dict['state'], 
                                     num_elements = num_elements, 
                                     center_index_re = state_center_re, 
                                     center_index_im = state_center_im, 
                                     sigma_re = state_sigma_re, 
                                     sigma_im = state_sigma_im, 
                                     magnitude_re = state_magnitude_re, 
                                     magnitude_im = state_magnitude_im,
                                     restrictions = state_restrictions)
        self.transition_tensor = initialize_transition_tensor(
            theta = self.thetas_dict['transition'],
            num_elements = num_elements,
            sigma_re_lo_lim = trans_sigma_re_lo_lim,
            sigma_re_hi_lim = trans_sigma_re_hi_lim,
            sigma_im = trans_sigma_im,
            mag_re = trans_magnitude_re,
            mag_im = trans_magnitude_im,
            restrict_dict = trans_restrictions
        )
        
        self.interaction_tensor = initialize_interaction_tensor(
            theta = self.thetas_dict['interaction'],
            num_elements = num_elements,
            sigma_re_lo_lim = inter_sigma_re_lo_lim,
            sigma_re_hi_lim = inter_sigma_re_hi_lim,
            sigma_im = inter_sigma_im,
            mag_re = inter_magnitude_re,
            mag_im = inter_magnitude_im,
            restrict_dict = inter_restrictions
        )
        
    def step_coil(self, renormalize = True, use_abs = False):
        self.transition_tensor, selected_subgroup = select_transition_tensor(self.state_tensor, self.transition_tensor, self.interaction_tensor, use_abs = use_abs)
        
        self.state_tensor = torch.matmul(self.transition_tensor,self.state_tensor)
        
        if renormalize:
            self.state_tensor = renormalize_to_unit_circle(self.state_tensor)
            
    def get_prob(self):
        return torch.real(self.state_tensor * torch.conj(self.state_tensor.sum()))