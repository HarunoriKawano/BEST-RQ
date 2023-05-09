import json

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from model import BestRqConfig, BestRqFramework


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
    encoder = ExampleEncoder()

    # `(batch size, time steps, feature size)`
    inputs = torch.randn(4, 1000, 80)
    # `(batch size)` Number of available time steps per batch
    input_lengths = torch.tensor([1000, 871, 389, 487])

    with open("config.json", "r", encoding="utf-8") as f:
        config = BestRqConfig(**json.load(f))

    model = BestRqFramework(config, encoder)

    targets, labels = model(inputs, input_lengths)
    loss = cross_entropy(targets, labels)

    print(loss)
