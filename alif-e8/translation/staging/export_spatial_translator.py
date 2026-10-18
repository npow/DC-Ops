"""Export the spatial FFN option separately; default is only full encoder stage0."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export as full_export
from prepare import ARTIFACTS, CALIBRATION, EVALUATION, REVISION, load_model
from translator import Translator
from spatial_transformer import SpatialEncoderLayer, SpatialEncoder, SpatialDecoder


def errors(actual, reference):
    delta = np.abs(actual-reference)
    return {"elements": int(delta.size), "mae": float(delta.mean()), "maximum_absolute_error": float(delta.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["encoder0", "encoder", "decoder", "both"], default="encoder0")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--quantize-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    folder = ARTIFACTS / "staging/spatial_translator"
    folder.mkdir(parents=True, exist_ok=True)
    full_export.ARTIFACTS = folder
    names = ["encoder", "decoder"] if args.model == "both" else [args.model]
    if not args.compile_only:
        model, tokenizer = load_model()
        translator = Translator(model, tokenizer)
        modules = {"encoder0": SpatialEncoderLayer(translator.encoder.layers[0]).eval()}
        if args.model != "encoder0":
            modules.update(encoder=SpatialEncoder(translator.encoder).eval(), decoder=SpatialDecoder(translator.decoder).eval())
        encoder_samples, decoder_samples = [], []
        with torch.no_grad():
            for text in CALIBRATION:
                if args.model == "encoder0":
                    encoder_samples.append(translator.encode_inputs(text))
                else:
                    result = translator.translate(text, collect=True)
                    encoder_samples.append(result["encoder_inputs"])
                    trace = result["decoder_inputs"]
                    decoder_samples.extend(trace[i] for i in sorted({0, min(1,len(trace)-1), len(trace)//2, len(trace)-1}))
            quantized = {name: full_export.quantize(name, modules[name], decoder_samples if name == "decoder" else encoder_samples)
                         for name in names}
            fixture = np.load(ARTIFACTS / "staging/encoder_0_cpu_fixture.npz")
            enc_args = tuple(torch.from_numpy(fixture[f"input_{i}"]) for i in range(2))
            audits = []
            for group, layers in [("encoder", translator.encoder.layers), ("decoder", translator.decoder.layers)]:
                for i, layer in enumerate(layers):
                    audits.append({"layer": f"{group}_{i}", "fc1_weight": list(layer.fc1.weight.shape),
                                   "fc2_weight": list(layer.fc2.weight.shape), "activation": "ReLU",
                                   "same_spatial_rewrite_applicable": layer.fc1.out_features == layer.fc2.in_features})
            report = {"revision": REVISION,
                      "samples_sha256": hashlib.sha256((ARTIFACTS / "samples.json").read_bytes()).hexdigest(),
                      "ffn_audit": audits, "calibration_count": len(encoder_samples), "hardware_tested": False}
            if args.model == "encoder0":
                original = translator.encoder.layers[0](*enc_args).numpy()
                floating = modules["encoder0"](*enc_args).numpy()
                actual = quantized["encoder0"](*enc_args).numpy()
                np.savez(folder / "encoder0_fixture.npz", input_0=fixture["input_0"], input_1=fixture["input_1"],
                         output=actual, fp32_output=original)
                report.update(float_vs_original=errors(floating, original), quantized_vs_original=errors(actual, original))
                rows = []
                for text in EVALUATION:
                    values = translator.encode_inputs(text)
                    rows.append({"input": text,
                                 "float_vs_original": errors(modules["encoder0"](*values).numpy(), translator.encoder.layers[0](*values).numpy()),
                                 "quantized_vs_original": errors(quantized["encoder0"](*values).numpy(), translator.encoder.layers[0](*values).numpy())})
                report.update(evaluation_denominator=len(rows), results=rows)
            elif args.model == "both":
                rows = []
                for text in EVALUATION:
                    expected = translator.translate(text)
                    actual = translator.translate(text, quantized["encoder"], quantized["decoder"])
                    rows.append({"input": text, "float": expected["text"], "spatial_quantized": actual["text"],
                                 "exact_match": expected["tokens"] == actual["tokens"], "finished": actual["finished"]})
                report.update(evaluation_denominator=len(rows), exact_matches=sum(row["exact_match"] for row in rows), results=rows)
            (folder / (args.model+"_evaluation.json")).write_text(json.dumps(report, indent=2)+"\n")
            print("HOST_REPORT", json.dumps(report), flush=True)
    if not args.quantize_only:
        for name in names:
            full_export.compile_model(name)
            metadata = json.loads((folder / (name+"_compile.json")).read_text())
            vela = np.load(folder / ("vela_"+name) / "output/out_vela.npz")
            metadata.update(scratch_bytes=int(vela["scratch_size"]), fast_scratch_bytes=int(vela["scratch_fast_size"]))
            metadata["fits_requested_budgets"] = (metadata["pte_bytes"] <= 2560*1024 and
                max(metadata["planned_buffers"]) <= 512*1024 and metadata["scratch_bytes"] <= 512*1024 and metadata["fast_scratch_bytes"] <= 384*1024)
            (folder / (name+"_memory.json")).write_text(json.dumps(metadata, indent=2)+"\n")
            print("MEMORY", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
