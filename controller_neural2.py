import torch
import torch.nn as nn

class NeuralController(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, output_size)  # Ensure this matches the output size of the action space

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))  # Output between -1 and 1 (if appropriate for your environment)

def get_weights(model):
    """Extract weights from a PyTorch model as NumPy arrays."""
    return [p.detach().numpy() for p in model.parameters()]

# ---- Load Weights Back into a Model ----
def set_weights(model, new_weights):
    """Update PyTorch model weights from a list of NumPy arrays."""
    for param, new_w in zip(model.parameters(), new_weights):
        param.data = torch.tensor(new_w, dtype=torch.float32)