"""Export independently staged U85 transformer layers and vocabulary shards."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import numpy as np
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export as full_export
from prepare import ARTIFACTS, CALIBRATION, EVALUATION, REVISION, load_model
from translator import Translator

OUTPUT = ARTIFACTS / "staging"
full_export.ARTIFACTS = OUTPUT


class DecoderBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.register_buffer("causal_mask", torch.triu(torch.full((1, 1, 32, 32), -100.0), diagonal=1))

    def forward(self, hidden, encoded, encoder_mask):
        return self.layer(hidden, encoded, self.causal_mask, encoder_mask)


class VocabularyShard(nn.Module):
    def __init__(self, decoder, start, end):
        super().__init__()
        self.head = nn.Linear(256, end-start, bias=True)
        self.head.weight = nn.Parameter(decoder.head.weight[start:end].detach().clone())
        self.head.bias = nn.Parameter(decoder.bias[0, start:end].detach().clone())

    def forward(self, hidden, last_position):
        return self.head((hidden * last_position).sum(dim=1))


class StagedTranslator:
    def __init__(self, encoders, decoders, heads):
        self.encoders, self.decoders, self.heads = encoders, decoders, heads

    def encoder(self, hidden, mask):
        for layer in self.encoders:
            hidden = layer(hidden, mask)
        return hidden

    def decoder(self, hidden, encoded, mask, last_position):
        for layer in self.decoders:
            hidden = layer(hidden, encoded, mask)
        return torch.cat([head(hidden, last_position) for head in self.heads], dim=-1)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--quantize-only", action="store_true")
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--spatial-ffn", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    OUTPUT = args.output.resolve() if args.output else ARTIFACTS / ("spatial_staging" if args.spatial_ffn else "staging")
    if args.spatial_ffn and OUTPUT.resolve() == (ARTIFACTS / "staging").resolve():
        raise ValueError("Spatial candidates must use a separate output directory")
    full_export.ARTIFACTS = OUTPUT
    torch.set_num_threads(4)
    torch.manual_seed(0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = [f"encoder_{i}" for i in range(6)] + [f"decoder_{i}" for i in range(2)] + [f"head_{i}" for i in range(4)]
    if args.spatial_ffn:
        if not args.compile_only:
            for name in names[:8]:
                for suffix in ("_u85.pte", "_compile.json"):
                    (OUTPUT / (name+suffix)).unlink(missing_ok=True)
        for name in names[8:]:
            for suffix in ("_a16w8.pt2", "_u85.pte", "_compile.json"):
                shutil.copyfile(ARTIFACTS / "staging" / (name+suffix), OUTPUT / (name+suffix))
            shutil.copytree(ARTIFACTS / "staging" / ("vela_"+name), OUTPUT / ("vela_"+name), dirs_exist_ok=True)
    if not args.compile_only:
        model, tokenizer = load_model()
        translator = Translator(model, tokenizer)
        if args.spatial_ffn:
            from spatial_transformer import SpatialEncoderLayer, SpatialDecoderLayer
            encoders = [SpatialEncoderLayer(layer).eval() for layer in translator.encoder.layers]
            decoders = [DecoderBlock(SpatialDecoderLayer(layer)).eval() for layer in translator.decoder.layers]
        else:
            encoders = list(translator.encoder.layers)
            decoders = [DecoderBlock(layer).eval() for layer in translator.decoder.layers]
        ranges = [(i*8000, (i+1)*8000 if i < 3 else 32001) for i in range(4)]
        heads = [VocabularyShard(translator.decoder, start, end).eval() for start, end in ranges]
        modules = encoders + decoders + heads
        samples = [[] for _ in modules]
        with torch.no_grad():
            for text in CALIBRATION:
                trace = translator.translate(text, collect=True)
                hidden, mask = trace["encoder_inputs"]
                for i, layer in enumerate(encoders):
                    samples[i].append((hidden.clone(), mask.clone()))
                    hidden = layer(hidden, mask)
                traces = trace["decoder_inputs"]
                for t in sorted({0, min(1, len(traces)-1), len(traces)//2, len(traces)-1}):
                    hidden, encoded, mask, last = traces[t]
                    for i, layer in enumerate(decoders):
                        samples[6+i].append((hidden.clone(), encoded.clone(), mask.clone()))
                        hidden = layer(hidden, encoded, mask)
                    for i in range(4):
                        samples[8+i].append((hidden.clone(), last.clone()))
        quantized = [torch.export.load(OUTPUT / (name+"_a16w8.pt2")).module() if args.spatial_ffn and name.startswith("head_")
                     else full_export.quantize(name, module, calibration)
                     for name, module, calibration in zip(names, modules, samples)]
        staged = StagedTranslator(quantized[:6], quantized[6:8], quantized[8:])
        results = []
        with torch.no_grad():
            for text in EVALUATION:
                baseline = translator.translate(text)
                result = translator.translate(text, staged.encoder, staged.decoder)
                row = {"input": text, "float": baseline["text"], "staged_a16w8": result["text"],
                       "exact_match": baseline["tokens"] == result["tokens"],
                       "float_tokens": baseline["tokens"], "staged_tokens": result["tokens"],
                       "finished": result["finished"]}
                print("EVALUATION", json.dumps(row), flush=True)
                results.append(row)
        report = {"denominator": len(results), "exact_matches": sum(r["exact_match"] for r in results),
                  "finished": sum(r["finished"] for r in results), "results": results,
                  "samples_sha256": hashlib.sha256((ARTIFACTS / "samples.json").read_bytes()).hexdigest(),
                  "vocabulary_ranges": ranges,
                  "calibration_counts": dict(zip(names, map(len, samples)))}
        if args.spatial_ffn:
            report["spatial_ffn"] = True
            report["checkpoint_revision"] = REVISION
            report["checkpoint_manifest_sha256"] = hashlib.sha256((ARTIFACTS / "checkpoint_manifest.json").read_bytes()).hexdigest()
            report["head_policy"] = "Original four head PT2E/PTE artifacts reused byte-for-byte"
            diagnostic = []
            with torch.no_grad():
                for text in ("T", "Hello"):
                    baseline = translator.translate(text)
                    actual = translator.translate(text, staged.encoder, staged.decoder)
                    diagnostic.append({"input": text, "float": baseline["text"], "spatial": actual["text"],
                                       "exact_match": baseline["tokens"] == actual["tokens"],
                                       "float_tokens": baseline["tokens"], "spatial_tokens": actual["tokens"],
                                       "finished": actual["finished"]})
            (OUTPUT / "diagnostic.json").write_text(json.dumps({"denominator": len(diagnostic), "results": diagnostic}, indent=2)+"\n")
        (OUTPUT / "evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
        print("EVALUATION_REPORT", json.dumps(report), flush=True)
    if not args.quantize_only:
        for name in (args.names or names):
            if not (OUTPUT / (name+"_compile.json")).exists():
                full_export.compile_model(name)
        results = {name: json.loads((OUTPUT / (name+"_compile.json")).read_text())
                   for name in names if (OUTPUT / (name+"_compile.json")).exists()}
        (OUTPUT / "manifest.json").write_text(json.dumps(results, indent=2)+"\n")
        if args.spatial_ffn:
            for name, metadata in results.items():
                vela = np.load(OUTPUT / ("vela_"+name) / "output/out_vela.npz")
                metadata.update(scratch_bytes=int(vela["scratch_size"]), fast_scratch_bytes=int(vela["scratch_fast_size"]))
                metadata["fits_requested_budgets"] = (metadata["pte_bytes"] <= 2560*1024 and
                    max(metadata["planned_buffers"]) <= 512*1024 and metadata["scratch_bytes"] <= 512*1024 and
                    metadata["fast_scratch_bytes"] <= 384*1024)
            (OUTPUT / "memory_manifest.json").write_text(json.dumps(results, indent=2)+"\n")
            if len(results) == len(names):
                import pack_pipeline
                pack_pipeline.ARTIFACTS = OUTPUT
                pack_pipeline.main()


if __name__ == "__main__":
    main()
