import torch

def complex_add(tensor):
    return torch.abs(torch.real(tensor)+torch.imag(tensor))

def select_transition_tensor(state_tensor,transition_tensor,interaction_tensor, sel_temperature):
    # Combine state and transition tensors into a norm_subgroups tensor
    norm_subgroups = torch.cat((state_tensor.unsqueeze(-1), transition_tensor), dim=2)
    
    # Get candidate transition tensors:
    candidate_transition_tensors = (torch.mul(interaction_tensor, norm_subgroups.unsqueeze(1).unsqueeze(1))).sum(-2) # [batches, states, states, states + 1]
    
    # Determine the largest and smallest states
    high_magnitude = torch.softmax((complex_add(state_tensor) / sel_temperature), dim = 1)
    low_magnitude = torch.softmax(1 - (complex_add(state_tensor) / sel_temperature), dim = 1)
    
    # Find which transition corresponds to moving from the highest state to the lowest state, and focus on that:
    transition_focus_slices = complex_add(torch.mul(torch.mul(candidate_transition_tensors, high_magnitude.unsqueeze(2).unsqueeze(1)), low_magnitude.unsqueeze(2).unsqueeze(2)))
    
    # Determine the selection weights by which slice is the highest in the focus transition
    selection_weights = torch.softmax(transition_focus_slices.sum(2).sum(1) / sel_temperature, dim = 1)
    
    # Rotate the candidate transition tensors to 1+0i so that we can do a weighted average
    rotated_candidate_transition_tensors = (candidate_transition_tensors * torch.conj(torch.sum(candidate_transition_tensors, dim = 1)))
    
    # Perform weighted averaging and rotate back to the original position (NOTE: we just use the first rotation here because they should all be the same in our case)
    selected_transition_tensor = torch.mul(rotated_candidate_transition_tensors, selection_weights.unsqueeze(1).unsqueeze(1)).sum(-1) / torch.conj(torch.sum(candidate_transition_tensors[:,:,:,0], dim = 1))
    
    return selected_transition_tensor