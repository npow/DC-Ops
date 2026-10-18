"""Execute the frozen 12-stage spatial translator through the host TOSA model."""
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torchao.quantization.pt2e.quantize_pt2e  # Register serialized quantization operators.
from executorch.backends.arm.common.pipeline_config import ArmPassPipelineConfig, SoftmaxDecompositionConfig
from executorch.backends.arm.test.runner_utils import TosaReferenceModelDispatch
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import ARTIFACTS, CALIBRATION, EVALUATION, REVISION, load_model
from translator import Translator
from export_staged import StagedTranslator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    folder = ARTIFACTS / "spatial_staging"
    samples = json.loads((ARTIFACTS / "samples.json").read_text())
    assert samples == {"calibration": CALIBRATION, "evaluation": EVALUATION}, "Frozen sample set changed"
    assert len(CALIBRATION) == 24 and len(EVALUATION) == 8
    previous = json.loads((folder / "evaluation.json").read_text())
    sample_hash = sha(ARTIFACTS / "samples.json")
    assert previous["samples_sha256"] == sample_hash
    assert previous["checkpoint_revision"] == REVISION
    names = [f"encoder_{i}" for i in range(6)] + [f"decoder_{i}" for i in range(2)] + [f"head_{i}" for i in range(4)]
    calibration_counts = previous["calibration_counts"]
    assert set(calibration_counts) == set(names)
    assert all(calibration_counts[name] == 24 for name in names[:6])
    assert all(calibration_counts[name] > 0 for name in names[6:])
    modules, reference_modules, lowering = [], [], []
    started = time.monotonic()
    for name in names:
        path = folder / (name + "_a16w8.pt2")
        before = sha(path)
        program = torch.export.load(path)
        modules.append(program.module())
        spec = TosaCompileSpec("TOSA-1.0+INT+int16+int4+cf")
        spec.set_pass_pipeline_config(ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.STABLE))
        began = time.monotonic()
        edge = to_edge_transform_and_lower(program, partitioner=[TOSAPartitioner(spec)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False))
        lowered = edge.exported_program().module()
        reference_modules.append(lowered)
        assert sha(path) == before, "Source PT2 artifact changed during evaluation"
        delegates = sum(node.op == "call_function" and "executorch_call_delegate" in str(node.target)
                        for node in lowered.graph.nodes)
        lowering.append({"name": name, "pt2_sha256": before, "delegates": delegates,
                         "lowering_seconds": time.monotonic() - began})
        print("LOWERED", json.dumps(lowering[-1]), flush=True)
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    quantized = StagedTranslator(modules[:6], modules[6:8], modules[8:])
    reference = StagedTranslator(reference_modules[:6], reference_modules[6:8], reference_modules[8:])
    report = {"execution": "Host executable TOSA reference model; no Vela command stream or E8 execution",
              "checkpoint_revision": REVISION, "checkpoint_manifest_sha256": sha(ARTIFACTS / "checkpoint_manifest.json"),
              "samples_sha256": sample_hash, "calibration_denominator": 24, "evaluation_denominator": 8,
              "calibration_counts": calibration_counts, "recalibrated": False, "lowered_once": lowering}
    results, diagnostics = [], []
    with torch.no_grad():
        for text in EVALUATION + ["T", "Hello"]:
            began = time.monotonic()
            baseline = translator.translate(text)
            host = translator.translate(text, quantized.encoder, quantized.decoder)
            with TosaReferenceModelDispatch():
                actual = translator.translate(text, reference.encoder, reference.decoder)
            row = {"input": text, "float_text": baseline["text"], "quantized_host_text": host["text"],
                   "tosa_text": actual["text"], "float_tokens": baseline["tokens"],
                   "quantized_host_tokens": host["tokens"], "tosa_tokens": actual["tokens"],
                   "tosa_matches_float_tokens": actual["tokens"] == baseline["tokens"],
                   "tosa_matches_quantized_tokens": actual["tokens"] == host["tokens"],
                   "float_eos": baseline["finished"], "quantized_host_eos": host["finished"],
                   "tosa_eos": actual["finished"], "host_evaluation_seconds": time.monotonic() - began}
            target = results if text in EVALUATION else diagnostics
            target.append(row)
            print("EVALUATED", json.dumps(row), flush=True)
            report.update(results=results, diagnostics=diagnostics, elapsed_seconds=time.monotonic()-started)
            (folder / "tosa_evaluation.partial.json").write_text(json.dumps(report, indent=2) + "\n")
    report.update(denominator=len(results), float_token_matches=sum(r["tosa_matches_float_tokens"] for r in results),
                  quantized_token_matches=sum(r["tosa_matches_quantized_tokens"] for r in results),
                  eos_finished=sum(r["tosa_eos"] for r in results), diagnostic_denominator=len(diagnostics),
                  diagnostic_float_token_matches=sum(r["tosa_matches_float_tokens"] for r in diagnostics),
                  diagnostic_quantized_token_matches=sum(r["tosa_matches_quantized_tokens"] for r in diagnostics),
                  diagnostic_eos_finished=sum(r["tosa_eos"] for r in diagnostics), elapsed_seconds=time.monotonic()-started)
    for item in lowering:
        assert sha(folder / (item["name"] + "_a16w8.pt2")) == item["pt2_sha256"], "Frozen source artifact changed"
    (folder / "tosa_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT", json.dumps({key: value for key, value in report.items()
                               if key not in ("results", "diagnostics", "lowered_once")}), flush=True)


if __name__ == "__main__":
    main()
