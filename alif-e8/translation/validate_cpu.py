"""Check that the portable ExecuTorch exports preserve the original translations."""
import hashlib
import json

import numpy as np
import torch
from executorch.runtime import Runtime
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

from prepare import ARTIFACTS, EVALUATION, load_model
from translator import Translator


def main():
    torch.set_num_threads(4)
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    programs = {name: Runtime.get().load_program(str(ARTIFACTS / (name+"_cpu_fp32.pte")))
                for name in ["encoder", "decoder"]}
    methods = {name: program.load_method("forward") for name, program in programs.items()}

    def encoder(*args):
        return methods["encoder"].execute(list(args))[0].clone()

    def decoder(*args):
        return methods["decoder"].execute(list(args))[0].clone()

    results = []
    with torch.no_grad():
        for text in EVALUATION:
            reference = translator.translate(text)
            actual = translator.translate(text, encoder, decoder)
            row = {"input": text, "reference": reference["text"], "executorch": actual["text"],
                   "exact_match": reference["tokens"] == actual["tokens"], "finished": actual["finished"]}
            results.append(row)
            print(json.dumps(row), flush=True)
        encoded_args = translator.encode_inputs("Please close the door")
        encoded = translator.encoder(*encoded_args)
        decoder_args = translator.decoder_inputs([model.config.decoder_start_token_id], encoded, encoded_args[1])
        data = {"encoder_input_0": encoded_args[0].numpy(), "encoder_input_1": encoded_args[1].numpy(),
                "encoder_fp32_output": encoded.numpy(), "decoder_fp32_output": translator.decoder(*decoder_args).numpy()}
        for i, arg in enumerate(decoder_args):
            data[f"decoder_input_{i}"] = arg.numpy()
        for name, args in [("encoder", encoded_args), ("decoder", decoder_args)]:
            quantized = torch.export.load(ARTIFACTS / (name+"_a16w8.pt2")).module()
            data[name+"_a16w8_output"] = quantized(*args).numpy()
        path = ARTIFACTS / "hardware_fixture.npz"
        np.savez(path, **data)
    report = {"denominator": len(results), "exact_matches": sum(r["exact_match"] for r in results),
              "results": results, "hardware_fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "execution_platform": "macOS host; correctness check only, not an E8 timing measurement"}
    (ARTIFACTS / "portable_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    assert report["exact_matches"] == len(results), report


if __name__ == "__main__":
    main()
