import torch
from torch import Tensor, nn
from zeta.nn import FeedForward, MultiQueryAttention
from torch.autograd import Variable
import math


def device_nvidia(num: int) -> torch.device:
    """
    Set device to GPU or CPU

    Args:
        num (int): GPU device number

    Returns:
        torch.device: Selected device
    """
    device = torch.device(f'cuda:{num}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device


class Transformer(nn.Module):
    """
    Transformer model with adaptive channel selection and position encoding

    A transformer-based model that processes multi-channel sequential data with:
    1. Adaptive channel selection using attention mechanism
    2. Positional encoding to capture sequence order
    3. Multiple transformer blocks for feature extraction
    4. Final classification layer

    Args:
        device (str): Device to run model on (cuda/cpu)
        num_tokens (int): Number of output classes
        dim (int): Model embedding dimension
        heads (int): Number of attention heads
        dim_head (int): Dimension of each attention head
        mult (int): Multiplier for FFN hidden dimension
        dropout (float): Dropout rate
        num_experts (int): Number of FFN experts
        depth (int): Number of transformer layers
    """

    def __init__(self,
                 device: str,
                 num_tokens: int,
                 dim: int,
                 heads: int,
                 dim_head: int = 64,
                 mult: int = 4,
                 dropout: float = 0.1,
                 num_experts: int = 3,
                 depth: int = 4):
        super().__init__()

        # Save parameters
        self.num_tokens = num_tokens
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout
        self.num_experts = num_experts
        self.depth = depth

        # Positional encoding
        self.pe = PositionalEncoding_signal(512, 8, dropout, device)

        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 8),
            nn.Sigmoid()
        )

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mult=mult,
                dropout=dropout,
                num_experts=num_experts
            ) for _ in range(depth)
        ])

        # Output classifier
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, num_tokens),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the model

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, seq_len)

        Returns:
            tuple[Tensor, Tensor]: Features and class probabilities
        """
        # Add positional encoding
        x = self.pe(x)

        # Apply channel attention and selection
        b, c, _ = x.size()
        channel_weights = self.channel_attention(x).view(b, c, 1)
        _, top_k_indices = channel_weights.squeeze(-1).topk(4, dim=-1)
        x = x * channel_weights
        x = torch.gather(x, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Get features and output probabilities
        features = x.view(x.size(0), -1)
        probs = self.to_out(features)

        return features, probs


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head attention and FFN

    Each block contains:
    1. Multi-head self attention layer
    2. Feed-forward network
    3. Layer normalization and residual connections

    Args:
        dim (int): Model dimension
        heads (int): Number of attention heads
        dim_head (int): Dimension of each attention head
        mult (int): FFN hidden dimension multiplier
        dropout (float): Dropout rate
        num_experts (int): Number of FFN experts
    """

    def __init__(self,
                 dim: int,
                 heads: int,
                 dim_head: int,
                 mult: int = 4,
                 dropout: float = 0.1,
                 num_experts: int = 3):
        super().__init__()

        # Save parameters
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout

        # Multi-head attention
        self.attn = MultiQueryAttention(dim, heads, qk_ln=True)

        # Feed-forward network
        self.ffn = FFN(
            dim=dim,
            hidden_dim=dim * mult,
            output_dim=dim,
            num_experts=num_experts
        )

        # Layer normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through transformer block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Processed tensor
        """
        # Attention with residual connection
        residual = x
        x, _, _ = self.attn(x)
        x = x + residual
        x = self.norm(x)

        # FFN with residual connection
        residual = x
        x = self.ffn(x)
        x = x + residual
        x = self.norm(x)

        return x


class FFN(nn.Module):
    """
    Feed-Forward Network layer

    Standard FFN layer with:
    1. Two linear transformations with ReLU activation
    2. Configurable hidden dimension

    Args:
        dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension
        num_experts (int): Number of parallel FFNs (unused in base version)
        capacity_factor (float): Capacity multiplier for experts
        mult (int): Hidden dimension multiplier
    """

    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_experts: int,
                 capacity_factor: float = 1.0,
                 mult: int = 4):
        super().__init__()

        # Save parameters
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult

        # Feed-forward network
        self.experts = FeedForward(dim, dim, mult)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through FFN

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Processed tensor
        """
        return self.experts(x)


class PositionalEncoding_signal(nn.Module):
    """
    Sinusoidal positional encoding for sequential signals

    Adds positional information to input features using:
    1. Sinusoidal encoding patterns
    2. Learnable dropout

    Args:
        leng (int): Encoding dimension length
        seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
        device (str): Device to store encodings
    """

    def __init__(self, leng: int, seq_len: int, dropout: float, device: str):
        super().__init__()

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Generate positional encodings
        pe = torch.zeros(seq_len, leng)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, leng, 2) * -(math.log(10000.0) / leng))

        pe[:, 0::2] = torch.sin(position * div_term) / math.sqrt(leng)
        pe[:, 1::2] = torch.cos(position * div_term) / math.sqrt(leng)

        self.pe = pe.unsqueeze(0).to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encodings to input

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Tensor with positional encodings added
        """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)