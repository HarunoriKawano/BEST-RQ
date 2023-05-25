import torch
from torch import nn
from torch.linalg import vector_norm

from model.config import Config


class RandomProjectionQuantizer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.random_projection = nn.Linear(
            config.input_feature_size * config.num_temporal_dimension_reduction_steps, config.code_book_size, bias=False
        )
        nn.init.xavier_uniform_(self.random_projection.weight)

        self.code_book = nn.Parameter(torch.randn(config.num_code_books, config.code_book_size))

        self.random_projection.weight.requires_grad = False
        self.code_book.requires_grad = False

    @torch.no_grad()
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

        labels = torch.argmin(vector_distances, dim=-1)

        return labels
