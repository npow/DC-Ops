"""Shape probes for diagnosing staged encoder hardware discrepancies."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export as full_export
from prepare import ARTIFACTS, CALIBRATION, load_model
from translator import Translator


class BatchedMatmul(nn.Module):
    def forward(self, q, k):
        return torch.matmul(q, k.transpose(-1, -2))


class SingleAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, hidden, mask):
        return self.attention(hidden, hidden, mask)


class MaskedSoftmax(nn.Module):
    def forward(self, scores, mask):
        return torch.softmax(scores + mask, dim=-1)


class FeedForward(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.fc1, self.fc2, self.norm = layer.fc1, layer.fc2, layer.norm2

    def forward(self, hidden):
        return self.norm(hidden + self.fc2(torch.relu(self.fc1(hidden))))


class AttentionResidualNorm(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.attention, self.norm = layer.attention, layer.norm1

    def forward(self, hidden, mask):
        return self.norm(hidden + self.attention(hidden, hidden, mask))


class FeedForwardBody(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.fc1, self.fc2 = layer.fc1, layer.fc2

    def forward(self, hidden):
        return self.fc2(torch.relu(self.fc1(hidden)))


class ResidualNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, hidden, delta):
        return self.norm(hidden + delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    output = ARTIFACTS / "staging/probes"
    output.mkdir(parents=True, exist_ok=True)
    full_export.ARTIFACTS = output
    attention = translator.encoder.layers[0].attention
    samples = [[] for _ in range(13)]
    with torch.no_grad():
        for text in CALIBRATION:
            hidden, mask = translator.encode_inputs(text)
            q = attention.q(hidden).reshape(1, 32, 8, 32).transpose(1, 2)
            k = attention.k(hidden).reshape(1, 32, 8, 32).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(32)
            residual = hidden + attention(hidden, hidden, mask)
            normed = translator.encoder.layers[0].norm1(residual)
            activated = torch.relu(translator.encoder.layers[0].fc1(normed))
            delta = translator.encoder.layers[0].fc2(activated)
            for target, values in zip(samples, [(hidden,), (q, k), (hidden, mask), (scores, mask), (residual,), (normed,), (hidden, mask), (normed,), (activated,), (normed,), (normed, delta), (normed,), (normed+delta,)]):
                target.append(tuple(x.clone() for x in values))
        hidden, mask = translator.encode_inputs("Please close the door")
        q = attention.q(hidden).reshape(1, 32, 8, 32).transpose(1, 2)
        k = attention.k(hidden).reshape(1, 32, 8, 32).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(32)
        residual = hidden + attention(hidden, hidden, mask)
        normed = translator.encoder.layers[0].norm1(residual)
        activated = torch.relu(translator.encoder.layers[0].fc1(normed))
        delta = translator.encoder.layers[0].fc2(activated)
        fixtures = [(hidden,), (q, k), (hidden, mask), (scores, mask), (residual,), (normed,), (hidden, mask), (normed,), (activated,), (normed,), (normed, delta), (normed,), (normed+delta,)]
        modules = [attention.q, BatchedMatmul(), SingleAttention(attention), MaskedSoftmax(), translator.encoder.layers[0].norm1, FeedForward(translator.encoder.layers[0]), AttentionResidualNorm(translator.encoder.layers[0]), translator.encoder.layers[0].fc1, translator.encoder.layers[0].fc2, FeedForwardBody(translator.encoder.layers[0]), ResidualNorm(translator.encoder.layers[0].norm2), nn.Sequential(translator.encoder.layers[0].fc1, nn.ReLU()), translator.encoder.layers[0].norm2]
        for name, module, calibration, values in zip(["linear32", "bmm", "attention", "softmax", "layernorm", "ffn", "attention_norm", "fc1", "fc2", "ffn_body", "residual_norm", "fc1_relu", "norm2"], modules, samples, fixtures):
            if args.names and name not in args.names:
                continue
            quantized = full_export.quantize(name, module.eval(), calibration)
            np.savez(output / (name+"_fixture.npz"),
                     **{f"input_{i}": t.numpy() for i, t in enumerate(values)},
                     output=quantized(*values).numpy(), fp32_output=module(*values).numpy())
            full_export.compile_model(name)


if __name__ == "__main__":
    main()
