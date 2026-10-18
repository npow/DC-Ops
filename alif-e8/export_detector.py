"""Export the trained DC-Ops box detector for the E8 Ethos-U85."""

import argparse
import hashlib
import json
import operator
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
REPO = "https://huggingface.co/datasets/abhijitbetigeri/dc-ops-dataset/resolve/main/"


class RawDetector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = model.model[:-1]
        self.head = model.model[-1]
        self.save = model.save

    def forward(self, image):
        saved = []
        x = image
        for layer in self.layers:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [
                    x if j == -1 else saved[j] for j in layer.f
                ]
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
        features = [saved[j] for j in self.head.f]
        return tuple(torch.cat((self.head.cv2[i](x), self.head.cv3[i](x)), 1)
                     for i, x in enumerate(features))


def letterbox(path, size):
    image = Image.open(path).convert("RGB")
    ratio = min(size / image.width, size / image.height)
    resized = image.resize((round(image.width * ratio), round(image.height * ratio)))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255


def calibration_images(folder, count):
    folder.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        "https://huggingface.co/api/datasets/abhijitbetigeri/dc-ops-dataset/tree/main/images?limit=1000"
    ) as response:
        entries = [x for x in json.load(response) if x["type"] == "file"]
    chosen = entries[::max(1, len(entries) // count)][:count]

    def download(entry):
        path = folder / Path(entry["path"]).name
        if not path.exists():
            urllib.request.urlretrieve(REPO + entry["path"], path)
        return path

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(download, chosen))


def integer_io(module):
    """Expose the calibrated INT8 boundary and keep float buffers off the MCU."""
    graph = module.graph
    placeholder = next(n for n in graph.nodes if n.op == "placeholder")
    users = list(placeholder.users)
    assert len(users) == 1 and "quantize_per_tensor" in str(users[0].target)
    quant = users[0]
    input_params = {"scale": quant.args[1], "zero_point": quant.args[2]}
    quant.replace_all_uses_with(placeholder)
    graph.erase_node(quant)
    output = next(n for n in graph.nodes if n.op == "output")
    params = []
    new_outputs = []
    for node in output.args[0]:
        assert "dequantize_per_tensor" in str(node.target), node
        params.append({"scale": node.args[1], "zero_point": node.args[2]})
        new_outputs.append(node.args[0])
    output.args = (tuple(new_outputs),)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return input_params, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--calibration-count", type=int, default=24)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    out = ROOT / "artifacts"
    checkpoint = out / "dc_ops_yolov8n_seg_v3.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        "5cfc0ae6c33390c6f401078250800da18b1ac48ce85a721de8612fd0ca08ccd9"
    )
    yolo = YOLO(checkpoint).model.float().eval().fuse()
    model = RawDetector(yolo).eval()
    paths = calibration_images(out / "calibration", args.calibration_count)
    samples = [letterbox(p, args.size) for p in paths]
    from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
    from executorch.backends.arm.quantizer import EthosUQuantizer
    from executorch.backends.arm.quantizer import get_symmetric_quantization_config
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    spec = EthosUCompileSpec(
        "ethos-u85-256", system_config="Ethos_U85_SRAM_MRAM", memory_mode="Shared_Sram",
        config_ini=str(ROOT / ".cmsis/ensemble_vela.ini"),
        extra_flags=["--verbose-performance", "--verbose-allocation"],
    )
    spec.dump_intermediate_artifacts_to(str(out / "vela"))
    print("Capturing detector", flush=True)
    graph = torch.export.export(model, (samples[0],), strict=False).module()
    # Export's runtime guards are not tensor operators and cannot be quantized.
    for node in list(graph.graph.nodes):
        if node.op == "call_function" and "_assert_tensor_metadata" in str(node.target):
            graph.graph.erase_node(node)
    graph.recompile()
    quantizer = EthosUQuantizer(spec).set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for i, image in enumerate(samples):
            prepared(image)
            print(f"Calibrated {i + 1}/{len(samples)}", flush=True)
    quantized = convert_pt2e(prepared)
    input_params, output_params = integer_io(quantized)
    integer_input = torch.clamp(torch.round(samples[0] / input_params["scale"]) +
                                input_params["zero_point"], -128, 127).to(torch.int8)
    with torch.no_grad():
        reference = model(samples[0])
        actual = quantized(integer_input)
    metadata = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "input_size": args.size, "input": input_params, "outputs": output_params,
        "runtime_input_layout": "NHWC storage; NCHW logical dimensions",
        "classes": yolo.names,
        "calibration": [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in paths],
        "output_shapes": [list(x.shape) for x in actual],
        "reference_mae": [float((r - (a.float() - p["zero_point"]) * p["scale"]).abs().mean())
                          for r, a, p in zip(reference, actual, output_params)],
    }
    (out / "export_metadata.json").write_text(json.dumps(metadata, indent=2))
    integer_input.numpy().tofile(out / "reference_input.bin")
    for i, tensor in enumerate(actual):
        tensor.numpy().tofile(out / f"reference_output_{i}.bin")
    print("Lowering for Ethos-U85", flush=True)
    exported = torch.export.export(quantized, (integer_input,), strict=False)
    edge = to_edge_transform_and_lower(
        exported, partitioner=[EthosUPartitioner(spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    calls = [str(n.target) for n in edge.exported_program().graph.nodes if n.op == "call_function"]
    print("Edge operators:", calls, flush=True)
    assert sum("executorch_call_delegate" in x for x in calls) == 1, calls
    assert all("executorch_call_delegate" in x or "getitem" in x for x in calls), calls
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    pte = bytes(program.buffer)
    (out / "dc_ops_u85.pte").write_bytes(pte)
    metadata["pte_bytes"] = len(pte)
    metadata["pte_sha256"] = hashlib.sha256(pte).hexdigest()
    metadata["planned_buffers"] = program.executorch_program.execution_plan[0].non_const_buffer_sizes
    (out / "export_metadata.json").write_text(json.dumps(metadata, indent=2))
    header = ROOT / "ai_layer/model_params.h"
    header.write_text(
        "#pragma once\n" + f"#define MODEL_INPUT_SIZE {args.size}\n" +
        f"#define MODEL_INPUT_SCALE {input_params['scale']:.10g}f\n" +
        f"#define MODEL_INPUT_ZERO_POINT {input_params['zero_point']}\n" +
        "static const float model_output_scales[] = {" +
        ",".join(f"{p['scale']:.10g}f" for p in output_params) + "};\n" +
        "static const int model_output_zero_points[] = {" +
        ",".join(str(p['zero_point']) for p in output_params) + "};\n"
    )
    with (ROOT / "ai_layer/model_pte.c").open("w") as f:
        f.write('#include "model_pte.h"\nconst unsigned long model_pte_size = ' + str(len(pte)) + ';\n')
        f.write('const unsigned char model_pte[] __attribute__((aligned(32))) = {\n')
        for offset in range(0, len(pte), 24):
            f.write(",".join(f"0x{x:02x}" for x in pte[offset:offset + 24]) + ",\n")
        f.write("};\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
