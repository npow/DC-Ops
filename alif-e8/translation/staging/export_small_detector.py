"""Export a smaller unchanged pretrained PP-OCR detector into separate artifacts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from executorch.backends.arm.ethosu import EthosUPartitioner
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_ocr import detector_input, spec_for, supported_activations
from ocr import CHECKPOINT, TextDetector, make_word
from prepare import ARTIFACTS, CALIBRATION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.width%32 or args.height%32:
        raise ValueError("The unchanged PP-OCR FPN requires width and height divisible32")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    folder = ARTIFACTS / f"small_detector_{args.width}x{args.height}"
    folder.mkdir(parents=True, exist_ok=True)
    captured = folder / "text_detector_a8w8.pt2"
    if not args.compile_only:
        detector = TextDetector().eval()
        checkpoint = CHECKPOINT / "en_ptocr_v3_det_infer.pth"
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = {key.replace("head.binarize.", "head."): value
                   for key, value in weights.items() if not key.startswith("head.thresh.")}
        detector.load_state_dict(weights)
        supported_activations(detector)
        samples, manifest = [], []
        for i, text in enumerate(CALIBRATION):
            scene = Image.new("RGB", (args.width, args.height), (240+i%3*5,)*3)
            word = make_word(text, 12+i%4*2)
            word.thumbnail((args.width-16, args.height//5))
            scene.paste(word, (6+i%3*2, 14+i%4*(args.height//5)))
            image_path = folder / f"calibration_{i:02d}.png"
            scene.save(image_path)
            samples.append((detector_input(scene),))
            manifest.append({"text": text, "image": image_path.name,
                             "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
        print("CAPTURE", args.width, args.height, flush=True)
        graph = torch.export.export(detector, samples[0], strict=False).module()
        for node in list(graph.graph.nodes):
            if (node.op == "call_module" and node.target == "_guards_fn") or (
                node.op == "call_function" and "_assert_tensor_metadata" in str(node.target)):
                graph.graph.erase_node(node)
        graph.recompile()
        quantizer = EthosUQuantizer(spec_for("ethos-u55-256")).set_global(get_symmetric_quantization_config())
        prepared = prepare_pt2e(graph, quantizer)
        for i, values in enumerate(samples):
            prepared(*values)
            print("CALIBRATE", i+1, "/", len(samples), flush=True)
        quantized = convert_pt2e(prepared)
        program = torch.export.export(quantized, samples[0], strict=False)
        torch.export.save(program, captured)
        np.savez(folder / "fixture.npz", input_0=samples[0][0].numpy(), output=quantized(*samples[0]).numpy())
        (folder / "calibration_manifest.json").write_text(json.dumps({
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "shape": [1, 3, args.height, args.width], "samples": manifest}, indent=2)+"\n")
    spec = spec_for("ethos-u55-256")
    spec.dump_intermediate_artifacts_to(str(folder / "vela"))
    print("COMPILE", flush=True)
    edge = to_edge_transform_and_lower(torch.export.load(captured), partitioner=[EthosUPartitioner(spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False))
    calls = [str(node.target) for node in edge.exported_program().graph.nodes if node.op == "call_function"]
    result = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    binary = bytes(result.buffer)
    (folder / "text_detector_u55_256.pte").write_bytes(binary)
    vela = np.load(folder / "vela/output/out_vela.npz")
    report = {"input_shape": [1, 3, args.height, args.width], "output_shape": [1, 1, args.height, args.width],
        "pte_bytes": len(binary), "sha256": hashlib.sha256(binary).hexdigest(),
        "planned_buffers": result.executorch_program.execution_plan[0].non_const_buffer_sizes,
        "scratch_bytes": int(vela["scratch_size"]), "fast_scratch_bytes": int(vela["scratch_fast_size"]),
        "delegates": sum("executorch_call_delegate" in call for call in calls),
        "cpu_ops": [call for call in calls if "executorch_call_delegate" not in call and "getitem" not in call],
        "target": "ethos-u55-256", "memory_mode": "Shared_Sram",
        "quality_status": "Unchanged pretrained weights at reduced image resolution; real camera quality not evaluated"}
    (folder / "compile.json").write_text(json.dumps(report, indent=2)+"\n")
    print("COMPILED", json.dumps(report), flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
