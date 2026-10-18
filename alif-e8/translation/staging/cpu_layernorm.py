"""Optional CPU LayerNorm contingency, retaining NPU matrix/attention work."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export as full_export
from prepare import ARTIFACTS, CALIBRATION, load_model
from translator import Translator


class WithoutLayerNorm(full_export.EthosUQuantizer):
    def set_global(self, configuration):
        super().set_global(configuration)
        self.set_module_type(nn.LayerNorm, None)
        return self


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = ARTIFACTS / "staging/cpu_layernorm"
    output.mkdir(parents=True, exist_ok=True)
    full_export.ARTIFACTS = output
    full_export.EthosUQuantizer = WithoutLayerNorm
    if not args.compile_only:
        model, tokenizer = load_model()
        translator = Translator(model, tokenizer)
        with torch.no_grad():
            calibration = [translator.encode_inputs(text) for text in CALIBRATION]
            quantized = full_export.quantize("encoder_0", translator.encoder.layers[0], calibration)
            inputs = translator.encode_inputs("Please close the door")
            np.savez(output / "encoder_0_fixture.npz",
                     **{f"input_{i}": x.numpy() for i, x in enumerate(inputs)},
                     output=quantized(*inputs).numpy())
    full_export.compile_model("encoder_0")


if __name__ == "__main__":
    main()
