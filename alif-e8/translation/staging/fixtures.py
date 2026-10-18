"""Frozen single-layer inputs and expected outputs for the board harness."""
import sys
from pathlib import Path

import numpy as np
import torch
import torchao.quantization.pt2e.quantize_pt2e

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import ARTIFACTS, load_model
from translator import Translator


def main():
    torch.set_num_threads(4)
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    destination = ARTIFACTS / "staging"
    data = {}
    with torch.no_grad():
        hidden, mask = translator.encode_inputs("Please close the door")
        for i in range(6):
            args = (hidden, mask)
            layer = torch.export.load(destination / f"encoder_{i}_a16w8.pt2").module()
            output = layer(*args)
            np.savez(destination / f"encoder_{i}_fixture.npz",
                     **{f"input_{j}": x.numpy() for j, x in enumerate(args)},
                     output=output.numpy())
            if i == 0:
                data = {"encoder_input_0": hidden.numpy(), "encoder_input_1": mask.numpy(),
                        "encoder_a16w8_output": output.numpy(),
                        "encoder_fp32_output": translator.encoder.layers[0](*args).numpy()}
            hidden = output
        encoded = hidden
        hidden, _, _, last = translator.decoder_inputs(
            [model.config.decoder_start_token_id], encoded, mask)
        for i in range(2):
            args = (hidden, encoded, mask)
            layer = torch.export.load(destination / f"decoder_{i}_a16w8.pt2").module()
            hidden = layer(*args)
            np.savez(destination / f"decoder_{i}_fixture.npz",
                     **{f"input_{j}": x.numpy() for j, x in enumerate(args)},
                     output=hidden.numpy())
        for i in range(4):
            layer = torch.export.load(destination / f"head_{i}_a16w8.pt2").module()
            output = layer(hidden, last)
            np.savez(destination / f"head_{i}_fixture.npz", input_0=hidden.numpy(),
                     input_1=last.numpy(), output=output.numpy())
    np.savez(destination / "hardware_fixture.npz", **data)


if __name__ == "__main__":
    main()
