import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableTemperatureSoftmax(nn.Module):
    def __init__(self, num_classes, initial_temperature=1.0):
        super(LearnableTemperatureSoftmax, self).__init__()
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, x):
        logits = x / self.temperature
        return F.softmax(logits, dim=-1)
    
class TemperatureSoftmax(nn.Module):
    def __init__(self, num_classes, temperature=1.0):
        super(TemperatureSoftmax, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature

    def forward(self, x):
        logits = x / self.temperature
        return F.softmax(logits, dim=-1)

class NeuralCoilLayer(nn.Module):
    def __init__(self, n_features, n_batch, device = "cpu"):
        super(NeuralCoilLayer, self).__init__()
        self.n_features = n_features
        self.attention_weights = nn.Linear(n_features, 1, bias=False)
        self.act = nn.SiLU()
        self.interaction_tensors = nn.Parameter(torch.randn(n_features, n_features, n_features, n_features + 1))
        self.topk_num = 1
        self.weightsoft = LearnableTemperatureSoftmax(n_features + 1, initial_temperature= 1.0)
        #self.weightsoft = TemperatureSoftmax(n_features + 1, temperature= 1.0)
        self.statesoft = LearnableTemperatureSoftmax(n_features, initial_temperature= 0.1)
        #self.statesoft = TemperatureSoftmax(n_features, temperature= 1 / n_features)
        
        starting_tensor = torch.softmax(torch.zeros(n_batch, n_features, n_features), dim = 1)
        if device == "cuda":
            self.starting_transition_tensor = starting_tensor.to("cuda")
        else:
            self.starting_transition_tensor = starting_tensor
        
    def step_coil(self, state_tensor, previous_transition_tensor):
        # Establish normalized subgroups by combining state tensor with transition tensors
        norm_subgroups = torch.cat((state_tensor.unsqueeze(-1), previous_transition_tensor), dim=2) # [batch_size, states, num_groups]
        batch_size, n_features, num_groups = norm_subgroups.shape
        
        # Compute scores for each normalized subgroup
        scores = self.act(self.attention_weights(norm_subgroups.permute(0,2,1))).sum(-1) # [batch_size, num_groups]
        
        weights = self.weightsoft(scores) # [batch_size, num_groups]
        
        selected_interaction_tensors = self.interaction_tensors # [states, states, states, states + 1]
        selected_norm_subgroups = norm_subgroups

        selected_transition_tensors = (torch.mul(selected_interaction_tensors, selected_norm_subgroups.unsqueeze(1).unsqueeze(1))).sum(-2) # [batches, states, states, states + 1]
        
        # We need a single transition tensor so we will average this as well
        selected_transition_tensor = (torch.mul(selected_transition_tensors, weights.unsqueeze(-2).unsqueeze(-2))).sum(-1) # [batches, states, states]

        # Generate state tensor from the transition tensor
        # Unsqueezing state_tensor to make it [batch_size, states, 1] for matrix multiplication
        # If we only want to use the state tensor
        selected_norm_tensor = norm_subgroups[:,:,0] # [batches, states]
        # If we want to use a mix of state and transition tensors
        #selected_norm_tensor = (torch.mul(norm_subgroups, weights.unsqueeze(1))).sum(-1) # [batches, states]
        
        selected_norm_tensor_unsqueezed = selected_norm_tensor.unsqueeze(2) # [batches, states, 1]

        # Performing batch matrix multiplication
        new_state_tensor_bmm = torch.bmm(selected_transition_tensor, selected_norm_tensor_unsqueezed) # [batches, states, 1]

        # Squeezing the result to get rid of the extra dimension
        new_state_tensor = new_state_tensor_bmm.squeeze(2) # [batches, states]
        
        softmax_tensor = self.statesoft(new_state_tensor)
        
        return softmax_tensor, selected_transition_tensor


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