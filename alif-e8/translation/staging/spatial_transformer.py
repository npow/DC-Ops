"""Equivalent FFNs that retain token rows on a spatial axis between convolutions."""
import torch
from torch import nn


class SpatialFFN(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.expand = nn.Conv2d(layer.fc1.in_features, layer.fc1.out_features, 1)
        self.project = nn.Conv2d(layer.fc2.in_features, layer.fc2.out_features, 1)
        self.expand.weight = nn.Parameter(layer.fc1.weight.detach().clone()[:, :, None, None])
        self.expand.bias = nn.Parameter(layer.fc1.bias.detach().clone())
        self.project.weight = nn.Parameter(layer.fc2.weight.detach().clone()[:, :, None, None])
        self.project.bias = nn.Parameter(layer.fc2.bias.detach().clone())

    def forward(self, hidden):
        hidden = hidden.transpose(1, 2).unsqueeze(-1)
        hidden = self.project(torch.relu(self.expand(hidden)))
        return hidden.squeeze(-1).transpose(1, 2).contiguous()


class SpatialEncoderLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.attention, self.norm1, self.norm2 = layer.attention, layer.norm1, layer.norm2
        self.ffn = SpatialFFN(layer)

    def forward(self, hidden, mask):
        hidden = self.norm1(hidden+self.attention(hidden, hidden, mask))
        return self.norm2(hidden+self.ffn(hidden))


class SpatialDecoderLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.attention, self.cross_attention = layer.attention, layer.cross_attention
        self.norm1, self.norm2, self.norm3 = layer.norm1, layer.norm2, layer.norm3
        self.ffn = SpatialFFN(layer)

    def forward(self, hidden, encoded, self_mask, encoder_mask):
        hidden = self.norm1(hidden+self.attention(hidden, hidden, self_mask))
        hidden = self.norm2(hidden+self.cross_attention(hidden, encoded, encoder_mask))
        return self.norm3(hidden+self.ffn(hidden))


class SpatialEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.layers = nn.ModuleList(SpatialEncoderLayer(layer) for layer in encoder.layers)

    def forward(self, hidden, mask):
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return hidden


class SpatialDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.layers = nn.ModuleList(SpatialDecoderLayer(layer) for layer in decoder.layers)
        self.head = decoder.head
        self.register_buffer("bias", decoder.bias)
        self.register_buffer("causal_mask", decoder.causal_mask)

    def forward(self, hidden, encoded, encoder_mask, last_position):
        for layer in self.layers:
            hidden = layer(hidden, encoded, self.causal_mask, encoder_mask)
        hidden = (hidden*last_position).sum(dim=1)
        return self.head(hidden)+self.bias
