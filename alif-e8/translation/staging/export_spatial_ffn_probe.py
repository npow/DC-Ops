"""Keep FFN token rows on a spatial axis across both 1x1 convolutions."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export as full_export
from prepare import ARTIFACTS, CALIBRATION, load_model
from translator import Translator


class SpatialFFN(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.expand = nn.Conv2d(256, 1536, 1)
        self.project = nn.Conv2d(1536, 256, 1)
        self.expand.weight = nn.Parameter(layer.fc1.weight.detach().clone()[:, :, None, None])
        self.expand.bias = nn.Parameter(layer.fc1.bias.detach().clone())
        self.project.weight = nn.Parameter(layer.fc2.weight.detach().clone()[:, :, None, None])
        self.project.bias = nn.Parameter(layer.fc2.bias.detach().clone())

    def forward(self, hidden):
        hidden = hidden.transpose(1, 2).unsqueeze(-1)
        hidden = self.project(torch.relu(self.expand(hidden)))
        return hidden.squeeze(-1).transpose(1, 2).contiguous()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    folder = ARTIFACTS / "staging/offline_spatial_ffn"
    folder.mkdir(parents=True, exist_ok=True)
    full_export.ARTIFACTS = folder
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    layer = translator.encoder.layers[0]
    spatial = SpatialFFN(layer).eval()
    with torch.no_grad():
        samples = []
        for text in CALIBRATION:
            hidden, mask = translator.encode_inputs(text)
            hidden = layer.norm1(hidden+layer.attention(hidden, hidden, mask))
            samples.append((hidden,))
        quantized = full_export.quantize("ffn_spatial", spatial, samples)
        fixture = np.load(ARTIFACTS / "staging/probes/ffn_body_fixture.npz")
        inputs = torch.from_numpy(fixture["input_0"])
        expected = spatial(inputs).numpy()
        actual = quantized(inputs).numpy()
        baseline = layer.fc2(torch.relu(layer.fc1(inputs))).numpy()
        np.savez(folder / "ffn_spatial_fixture.npz", input_0=inputs.numpy(), output=actual, fp32_output=baseline)
    full_export.compile_model("ffn_spatial")
    metadata = json.loads((folder / "ffn_spatial_compile.json").read_text())
    vela = np.load(folder / "vela_ffn_spatial/output/out_vela.npz")
    metadata.update(scratch_bytes=int(vela["scratch_size"]), fast_scratch_bytes=int(vela["scratch_fast_size"]),
                    input_shape=vela["input_shape"].tolist(), output_shape=vela["output_shape"].tolist(),
                    float_equivalence_max_error=float(np.abs(expected-baseline).max()),
                    host_quantized_mae=float(np.abs(actual-baseline).mean()),
                    host_quantized_max_error=float(np.abs(actual-baseline).max()),
                    hardware_tested=False,
                    intent="Avoid treating FFN output rows as width then reinterpreting them as batch at the second linear")
    (folder / "report.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
