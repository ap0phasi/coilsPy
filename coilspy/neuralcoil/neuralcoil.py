import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCoilLayer(nn.Module):
    def __init__(self, n_features, n_batch, sel_temp = 1e-3, norm_temp = 7, device = "cpu"):
        super(NeuralCoilLayer, self).__init__()
        self.interaction_tensors = nn.Parameter(torch.rand(n_features, n_features, n_features, n_features + 1))
        self.sel_temp = sel_temp
        self.norm_temp = norm_temp
        
        starting_tensor = torch.softmax(torch.zeros(n_batch, n_features, n_features), dim = 1)
        if device == "cuda":
            self.starting_transition_tensor = starting_tensor.to("cuda")
        else:
            self.starting_transition_tensor = starting_tensor
            
    def select_transition_tensor(self, state_tensor,transition_tensor, interaction_tensor, sel_temperature):
        # Combine state and transition tensors into a norm_subgroups tensor
        norm_subgroups = torch.cat((state_tensor.unsqueeze(-1), transition_tensor), dim=2)
        
        # Get candidate transition tensors:
        candidate_transition_tensors = (torch.mul(interaction_tensor, norm_subgroups.unsqueeze(1).unsqueeze(1))).sum(-2) # [batches, states, states, states + 1]
        
        # Determine the largest and smallest states
        high_magnitude = torch.softmax(((state_tensor) / sel_temperature), dim = 1)
        low_magnitude = torch.softmax(1 - ((state_tensor) / sel_temperature), dim = 1)
        
        # Find which transition corresponds to moving from the highest state to the lowest state, and focus on that:
        transition_focus_slices = (torch.mul(torch.mul(candidate_transition_tensors, high_magnitude.unsqueeze(2).unsqueeze(1)), low_magnitude.unsqueeze(2).unsqueeze(2)))
        
        # Determine the selection weights by which slice is the highest in the focus transition
        selection_weights = torch.softmax(transition_focus_slices.sum(2).sum(1) / sel_temperature, dim = 1)
        
        # Perform weighted averaging
        selected_transition_tensor = torch.mul(candidate_transition_tensors, selection_weights.unsqueeze(1).unsqueeze(1)).sum(-1)
        
        return selected_transition_tensor
        
    def step_coil(self, state_tensor, transition_tensor):
        
        selected_transition_tensor = self.select_transition_tensor(state_tensor, transition_tensor, self.interaction_tensors, sel_temperature= self.sel_temp)
        
        selected_transition_tensor = torch.softmax(selected_transition_tensor * self.norm_temp, dim =1)
        
        new_state_tensor = torch.mul(selected_transition_tensor, state_tensor.unsqueeze(1)).sum(dim = -1)
        

        return new_state_tensor, selected_transition_tensor


    def forward(self, x):
        batch, length, n_features = x.size()
        output = x.new_empty(batch, length, n_features)

        # Initialize previous transition tensors (for the first step)
        # Assuming it's a list of zero tensors for simplicity
        transition_tensor = self.starting_transition_tensor

        for l in range(length):
            state_tensor = x[:, l, :]
            
            # Compute output for this step
            output[:, l, :], transition_tensor = self.step_coil(state_tensor, transition_tensor)

        return output, transition_tensor