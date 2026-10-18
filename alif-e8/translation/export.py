"""Calibrate, evaluate, and compile the fixed-size translator for Ethos-U85."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from executorch.backends.arm.common.pipeline_config import ArmPassPipelineConfig, SoftmaxDecompositionConfig
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_a16w8_quantization_config
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from prepare import ARTIFACTS, CALIBRATION, EVALUATION, load_model
from translator import Translator


def compile_spec():
    spec = EthosUCompileSpec(
        "ethos-u85-256", system_config="Ethos_U85_SRAM_OSPI", memory_mode="Dedicated_Sram",
        config_ini=str(Path(__file__).resolve().parent.parent / ".cmsis/ensemble_vela.ini"),
        extra_flags=["--verbose-performance", "--verbose-allocation"],
    )
    spec.set_pass_pipeline_config(ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.STABLE))
    return spec


def quantize(name, model, samples):
    print("CAPTURE", name, flush=True)
    graph = torch.export.export(model, samples[0], strict=False).module()
    for node in list(graph.graph.nodes):
        if (node.op == "call_module" and node.target == "_guards_fn") or (
            node.op == "call_function" and "_assert_tensor_metadata" in str(node.target)
        ):
            graph.graph.erase_node(node)
    graph.recompile()
    quantizer = EthosUQuantizer(compile_spec()).set_global(get_symmetric_a16w8_quantization_config())
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for i, args in enumerate(samples):
            prepared(*args)
            if i % 8 == 0:
                print("CALIBRATE", name, i+1, "/", len(samples), flush=True)
    quantized = convert_pt2e(prepared)
    program = torch.export.export(quantized, samples[0], strict=False)
    torch.export.save(program, ARTIFACTS / (name + "_a16w8.pt2"))
    return quantized


def compile_model(name):
    program = torch.export.load(ARTIFACTS / (name + "_a16w8.pt2"))
    spec = compile_spec()
    spec.dump_intermediate_artifacts_to(str(ARTIFACTS / ("vela_" + name)))
    print("LOWER", name, flush=True)
    edge = to_edge_transform_and_lower(
        program, partitioner=[EthosUPartitioner(spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    calls = [str(n.target) for n in edge.exported_program().graph.nodes if n.op == "call_function"]
    print("EDGE OPERATORS", name, calls, flush=True)
    output = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    pte = bytes(output.buffer)
    path = ARTIFACTS / (name + "_u85.pte")
    path.write_bytes(pte)
    result = {"pte_bytes": len(pte), "sha256": hashlib.sha256(pte).hexdigest(),
              "planned_buffers": output.executorch_program.execution_plan[0].non_const_buffer_sizes,
              "delegates": sum("executorch_call_delegate" in x for x in calls),
              "cpu_ops": [x for x in calls if "executorch_call_delegate" not in x and "getitem" not in x]}
    (ARTIFACTS / (name + "_compile.json")).write_text(json.dumps(result, indent=2)+"\n")
    print("COMPILED", name, json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--quantize-only", action="store_true")
    parser.add_argument("--model", choices=["encoder", "decoder", "both"], default="both")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    started = time.time()
    if not args.compile_only:
        model, tokenizer = load_model()
        translator = Translator(model, tokenizer)
        encoder_samples, decoder_samples = [], []
        for text in CALIBRATION:
            result = translator.translate(text, collect=True)
            encoder_samples.append(result["encoder_inputs"])
            traces = result["decoder_inputs"]
            decoder_samples.extend(traces[i] for i in sorted({0, min(1,len(traces)-1), len(traces)//2, len(traces)-1}))
        encoder = quantize("encoder", translator.encoder, encoder_samples)
        decoder = quantize("decoder", translator.decoder, decoder_samples)
        results = []
        with torch.no_grad():
            for text in EVALUATION:
                baseline = translator.translate(text)
                quantized = translator.translate(text, encoder, decoder)
                row = {"input": text, "float": baseline["text"], "a16w8": quantized["text"],
                       "exact_match": baseline["tokens"] == quantized["tokens"],
                       "finished": quantized["finished"]}
                results.append(row)
                print("EVALUATION", json.dumps(row), flush=True)
        samples = ARTIFACTS / "samples.json"
        report = {"denominator": len(results), "exact_matches": sum(r["exact_match"] for r in results),
                  "finished": sum(r["finished"] for r in results), "results": results,
                  "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
                  "elapsed_seconds": time.time()-started}
        (ARTIFACTS / "quantization_evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
        print("QUANTIZATION", json.dumps(report), flush=True)
    if not args.quantize_only:
        for name in ([args.model] if args.model != "both" else ["encoder", "decoder"]):
            compile_model(name)


if __name__ == "__main__":
    main()
