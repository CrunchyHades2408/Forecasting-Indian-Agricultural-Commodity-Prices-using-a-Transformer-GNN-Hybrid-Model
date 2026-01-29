import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_dim = d_model

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        x = self.input_projection(x)
        x = self.transformer(x)

        # Use last timestep representation
        return x[:, -1, :]
