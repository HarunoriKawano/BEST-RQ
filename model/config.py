from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    mask_prob: float  # 0.0 - 1.0
    mask_time: float  # Mask time sec (Default: 0.4)
    input_feature_size: int  # Dimension of input.
    stride_time: float  # stride_time sec.
    code_book_size: int  # Dimension of code book (Default: 16)
    num_code_books: int  # Number of code books (Default: 8192)
    num_temporal_dimension_reduction_steps: int  # Number of temporal dimension reduction steps by the encoder
    encoder_hidden_size: int  # Number of encoder output dimensions
