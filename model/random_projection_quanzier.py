import torch
from torch import nn
from torch.linalg import vector_norm

from best_rq_config import BestRqConfig


class RandomProjectionQuantizer(nn.Module):
    def __init__(self, config: BestRqConfig):
        super().__init__()
        self.mask_size = int(config.quantized_representation_time / config.stride_time)
        self.random_projection = nn.Linear(config.mel_filter_size*self.mask_size, config.code_book_size, bias=False)
        nn.init.xavier_uniform_(self.random_projection.weight)

        self.code_book = nn.Parameter(torch.randn(config.num_code_books, config.code_book_size))

    def forward(self, input_values: torch.Tensor, mask_time_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_values (torch.Tensor): with shape `(B, L, D)`
            mask_time_indices (torch.Tensor): with shape `(B, L)`

        Returns:
            torch.Tensor with shape `(N)`

        """
        targets = self.random_projection(input_values[mask_time_indices == 1]).unsqueeze(1)

        # Compute l2 norm targets and code vectors
        vector_distances = vector_norm(targets - self.code_book, dim=-1)

        labels = torch.argmin(vector_distances)

        return labels
