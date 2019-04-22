import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, num_of_neurones, num_of_hidden_layers, num_of_inputs, num_of_outputs):
        super(Net, self).__init__()
        self.num_of_hidden_layers = num_of_hidden_layers
        self.input_linear = nn.Linear(num_of_inputs  , num_of_neurones)
        self.linears = nn.ModuleList([nn.Linear(num_of_neurones, num_of_neurones) for i in range(num_of_hidden_layers)])
        self.output_linear = nn.Linear(num_of_neurones, num_of_outputs)


    def forward(self, x):
        x = F.relu(self.input_linear(x))
        for i, _ in enumerate(self.linears):
            x = self.linears[i](x)
        x = self.output_linear(x)
        x = torch.sigmoid(x)
        # x = x.view(-1, 8)
        return x