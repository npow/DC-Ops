"""Compile the frozen encoder0 three-part contingency without changing live assets."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchao.quantization.pt2e.quantize_pt2e

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export as full_export
from prepare import ARTIFACTS


def metrics(actual, expected):
    absolute = np.abs(actual-expected)
    return {"elements": int(actual.size), "mae": float(absolute.mean()),
            "maximum_absolute_error": float(absolute.max()),
            "per_token_mae": absolute.mean(-1).flatten().tolist()}


def main():
    torch.set_num_threads(4)
    staging = ARTIFACTS / "staging"
    source = staging / "split_ffn"
    output = staging / "offline_encoder0_split"
    output.mkdir(parents=True, exist_ok=True)
    full_export.ARTIFACTS = output
    fixture = np.load(staging / "encoder_0_cpu_fixture.npz")
    embeddings, mask = [torch.from_numpy(fixture[f"input_{i}"]) for i in range(2)]
    names = [f"encoder_0_{part}" for part in ("attention", "expand", "project")]
    modules = {}
    for name in names:
        filename = name+"_a16w8.pt2"
        shutil.copyfile(source / filename, output / filename)
        modules[name] = torch.export.load(output / filename).module()
    with torch.no_grad():
        attention = modules[names[0]](embeddings, mask)
        expansion = modules[names[1]](attention)
        projected = modules[names[2]](attention, expansion)
        original = torch.export.load(staging / "encoder_0_a16w8.pt2").module()(embeddings, mask)
    for name, inputs, result in [(names[0], [embeddings, mask], attention),
                                 (names[1], [attention], expansion),
                                 (names[2], [attention, expansion], projected)]:
        np.savez(output / (name+"_fixture.npz"),
                 **{f"input_{i}": value.numpy() for i, value in enumerate(inputs)}, output=result.numpy())
        full_export.compile_model(name)
    np.savez(output / "assembled_fixture.npz", input_0=embeddings.numpy(), input_1=mask.numpy(),
             output=projected.numpy(), fp32_output=fixture["output"], original_quantized_output=original.numpy())
    report = {"description": "Offline alternative: attention+residual+norm, FC1+ReLU, FC2+residual+norm",
              "frozen_input": "Please close the door", "hardware_tested": False,
              "split_host_vs_fp32": metrics(projected.numpy(), fixture["output"]),
              "original_quantized_host_vs_fp32": metrics(original.numpy(), fixture["output"]),
              "split_host_vs_original_quantized_host": metrics(projected.numpy(), original.numpy()),
              "full30_part_host_evaluation": json.loads((source / "evaluation.json").read_text()),
              "parts": {}}
    for name in names:
        compiled = json.loads((output / (name+"_compile.json")).read_text())
        vela = np.load(output / ("vela_"+name) / "output/out_vela.npz")
        compiled.update(scratch_bytes=int(vela["scratch_size"]), fast_scratch_bytes=int(vela["scratch_fast_size"]))
        report["parts"][name] = compiled
    hardware = ARTIFACTS / "repaired_encoder0_output.bin"
    if hardware.exists():
        actual = np.fromfile(hardware, dtype="<f4").reshape(fixture["output"].shape)
        report["repaired_original_hardware_vs_fp32"] = metrics(actual, fixture["output"])
        report["repaired_original_hardware_vs_quantized_host"] = metrics(actual, original.numpy())
        report["hardware_capture_sha256"] = hashlib.sha256(hardware.read_bytes()).hexdigest()
    report["assessment"] = (
        "Large row-dependent hardware error persists after weight-storage repair, while host A16W8 matches FP32 closely. "
        "Previously tested isolated FC1+ReLU, FC2, residual+norm, attention, and LayerNorm were close to host references. "
        "The new boundary separates the two wide FFN convolutions into distinct NPU invocations. "
        "This targets fused-chain behavior but is not yet a demonstrated hardware accuracy fix. "
        "Live user testing and the live pipeline assets remain unchanged.")
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "full30_part_host_evaluation"}, indent=2))


if __name__ == "__main__":
    main()
