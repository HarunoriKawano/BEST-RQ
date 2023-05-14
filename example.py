import json

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from model import Config, BestRqFramework


class ExampleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(80, 192)
        self.num_temporal_dimension_reduction_steps = 4

    def forward(self, inputs, _):
        batch_size, length, _ = inputs.size()

        hidden_states = self.linear(inputs)
        hidden_states = hidden_states.view(batch_size, length // self.num_temporal_dimension_reduction_steps, -1)

        return hidden_states


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Use device: {device}")

    encoder = ExampleEncoder().to(device)

    # `(batch size, time steps, feature size)`
    inputs = torch.rand(4, 1000, 80).to(device)
    # `(batch size)` Number of available time steps per batch
    input_lengths = torch.tensor([1000, 871, 389, 487]).to(device)

    with open("config.json", "r", encoding="utf-8") as f:
        config = Config(**json.load(f))

    model = BestRqFramework(config, encoder).to(device)

    targets, labels = model(inputs, input_lengths)
    loss = cross_entropy(targets, labels)

    print(loss)
    loss.backward()
