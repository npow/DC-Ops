"""Calibrate PP-OCR and compile its stages for the E8's U55 accelerators."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from ocr import SOURCE, decode, load_ocr, make_word, recognition_input
from prepare import ARTIFACTS, CALIBRATION, EVALUATION
from pytorchocr.modeling.backbones.rec_lcnetv3 import LearnableRepLayer


class SupportedActivation(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, value):
        if self.kind == "Hswish":
            return torch.nn.functional.hardswish(value)
        scaled = value * 1.2 if self.kind == "Hsigmoid" else value
        return (torch.relu(scaled + 3.0) - torch.relu(scaled - 3.0)) / 6.0


def supported_activations(model):
    for name, child in list(model.named_children()):
        if type(child).__name__ in ("Hswish", "Hsigmoid", "Hardsigmoid"):
            model._modules[name] = SupportedActivation(type(child).__name__)
        else:
            supported_activations(child)


def spec_for(target):
    return EthosUCompileSpec(target, system_config="RTSS_HP_SRAM_OSPI", memory_mode="Shared_Sram",
        config_ini=str(Path(__file__).resolve().parent.parent / ".cmsis/ensemble_vela.ini"),
        extra_flags=["--verbose-performance"])


def detector_input(image):
    pixels = torch.from_numpy(np.asarray(image)[:, :, ::-1].copy()).permute(2,0,1).float()/255
    return ((pixels-torch.tensor([.485,.456,.406])[:,None,None]) / torch.tensor([.229,.224,.225])[:,None,None]).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    samples = {}
    if not args.compile_only:
        detector, recognizer, characters = load_ocr()
        reference = [decode(recognizer(recognition_input(make_word(t))), characters) for t in EVALUATION]
        for module in list(recognizer.modules()):
            if isinstance(module, LearnableRepLayer):
                module.rep()
        actual = [decode(recognizer(recognition_input(make_word(t))), characters) for t in EVALUATION]
        assert actual == reference, (actual, reference)
        folder = ARTIFACTS / "ocr_calibration"
        folder.mkdir(exist_ok=True)
        rec_samples, det_samples = [], []
        manifest = []
        for i, text in enumerate(CALIBRATION):
            word = make_word(text, 24+i%4*4)
            path = folder / f"word_{i:02d}.png"
            word.save(path)
            manifest.append({"file": path.name, "text": text, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            rec_samples.append((recognition_input(word),))
            scene = Image.new("RGB", (288,480), (240+i%3*5,)*3)
            word.thumbnail((260,100))
            scene.paste(word, (12+i%3*4, 30+i%4*100))
            det_samples.append((detector_input(scene),))
        (ARTIFACTS / "ocr_calibration_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        quantized = {}
        for name, model, inputs in [("text_detector",detector,det_samples), ("text_recognizer",recognizer,rec_samples)]:
            print("CAPTURE",name,flush=True)
            expected_output = model(*inputs[0]).detach().clone()
            supported_activations(model)
            torch.testing.assert_close(model(*inputs[0]), expected_output, atol=1e-4, rtol=1e-4)
            graph = torch.export.export(model, inputs[0], strict=False).module()
            for node in list(graph.graph.nodes):
                if (node.op == "call_module" and node.target == "_guards_fn") or (
                    node.op == "call_function" and "_assert_tensor_metadata" in str(node.target)):
                    graph.graph.erase_node(node)
            graph.recompile()
            quantizer = EthosUQuantizer(spec_for("ethos-u55-256")).set_global(get_symmetric_quantization_config())
            prepared = prepare_pt2e(graph,quantizer)
            with torch.no_grad():
                for i, value in enumerate(inputs):
                    prepared(*value)
                    if i%8 == 0: print("CALIBRATE",name,i+1,flush=True)
            module = convert_pt2e(prepared)
            program = torch.export.export(module, inputs[0], strict=False)
            torch.export.save(program,ARTIFACTS/(name+"_a8w8.pt2"))
            quantized[name] = module
            samples[name] = inputs[0]
        rows = []
        with torch.no_grad():
            for text, expected in zip(EVALUATION,reference):
                image = make_word(text)
                width = min(320, math.ceil(48*image.width/image.height))
                value = decode(quantized["text_recognizer"](recognition_input(image)),characters,width)
                row = {"input":text,"float":expected,"int8":value,"matches_float":value==expected,"matches_text":value==text}
                rows.append(row)
                print("OCR_EVAL",json.dumps(row),flush=True)
        report = {"denominator":len(rows),"matches_float":sum(r["matches_float"] for r in rows),
                  "exact_text_matches":sum(r["matches_text"] for r in rows),"results":rows,
                  "scope":"synthetic printed-text smoke test; not real-camera accuracy",
                  "padding_policy":"Only decode timesteps inside the resized crop width; exclude artificial right padding"}
        (ARTIFACTS/"ocr_quantization_evaluation.json").write_text(json.dumps(report,indent=2)+"\n")
        if report["matches_float"] < len(rows)-1:
            raise RuntimeError("OCR quantization changed too many smoke-test predictions")
    for name,target in [("text_detector","ethos-u55-256"),("text_recognizer","ethos-u55-128")]:
        print("COMPILE",name,target,flush=True)
        program = torch.export.load(ARTIFACTS/(name+"_a8w8.pt2"))
        spec = spec_for(target)
        spec.dump_intermediate_artifacts_to(str(ARTIFACTS/("vela_"+name)))
        edge = to_edge_transform_and_lower(program, partitioner=[EthosUPartitioner(spec)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False))
        calls = [str(n.target) for n in edge.exported_program().graph.nodes if n.op=="call_function"]
        output = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
        path = ARTIFACTS/(name+"_"+target+".pte")
        path.write_bytes(bytes(output.buffer))
        report = {"target":target,"pte_bytes":path.stat().st_size,
                  "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                  "planned_buffers":output.executorch_program.execution_plan[0].non_const_buffer_sizes,
                  "delegates":sum("executorch_call_delegate" in x for x in calls),
                  "cpu_ops":[x for x in calls if "executorch_call_delegate" not in x and "getitem" not in x]}
        (ARTIFACTS/(name+"_compile.json")).write_text(json.dumps(report,indent=2)+"\n")
        print("COMPILED",name,json.dumps(report),flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
