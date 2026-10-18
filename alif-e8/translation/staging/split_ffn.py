"""Keep each FFN expansion and projection in separate NPU invocations."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export as full_export
from prepare import ARTIFACTS, CALIBRATION, EVALUATION, load_model
from translator import Translator

OUTPUT = ARTIFACTS / "staging/split_ffn"


class EncoderAttention(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.attention, self.norm = layer.attention, layer.norm1

    def forward(self, hidden, mask):
        return self.norm(hidden + self.attention(hidden, hidden, mask))


class DecoderSelfAttention(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.attention, self.norm = layer.attention, layer.norm1
        self.register_buffer("mask", torch.triu(torch.full((1, 1, 32, 32), -100.0), diagonal=1))

    def forward(self, hidden):
        return self.norm(hidden + self.attention(hidden, hidden, self.mask))


class DecoderCrossAttention(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.attention, self.norm = layer.cross_attention, layer.norm2

    def forward(self, hidden, encoded, mask):
        return self.norm(hidden + self.attention(hidden, encoded, mask))


class ProjectionResidualNorm(nn.Module):
    def __init__(self, fc2, norm):
        super().__init__()
        self.fc2, self.norm = fc2, norm

    def forward(self, hidden, activated):
        return self.norm(hidden + self.fc2(activated))


class SplitTranslator:
    def __init__(self, modules):
        self.modules = modules

    def encoder(self, hidden, mask):
        for i in range(6):
            hidden = self.modules[f"encoder_{i}_attention"](hidden, mask)
            activated = self.modules[f"encoder_{i}_expand"](hidden)
            hidden = self.modules[f"encoder_{i}_project"](hidden, activated)
        return hidden

    def decoder(self, hidden, encoded, mask, last):
        for i in range(2):
            hidden = self.modules[f"decoder_{i}_self"](hidden)
            hidden = self.modules[f"decoder_{i}_cross"](hidden, encoded, mask)
            activated = self.modules[f"decoder_{i}_expand"](hidden)
            hidden = self.modules[f"decoder_{i}_project"](hidden, activated)
        return torch.cat([self.modules[f"head_{i}"](hidden, last) for i in range(4)], -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-probe", action="store_true")
    parser.add_argument("--quantize-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--names", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full_export.ARTIFACTS = OUTPUT
    if args.compile_only:
        names = args.names or [p.name.removesuffix("_a16w8.pt2") for p in sorted(OUTPUT.glob("*_a16w8.pt2"))]
        for name in names:
            full_export.compile_model(name)
        return
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    modules = {}
    for i, layer in enumerate(translator.encoder.layers):
        modules[f"encoder_{i}_attention"] = EncoderAttention(layer)
        modules[f"encoder_{i}_expand"] = nn.Sequential(layer.fc1, nn.ReLU())
        modules[f"encoder_{i}_project"] = ProjectionResidualNorm(layer.fc2, layer.norm2)
    for i, layer in enumerate(translator.decoder.layers):
        modules[f"decoder_{i}_self"] = DecoderSelfAttention(layer)
        modules[f"decoder_{i}_cross"] = DecoderCrossAttention(layer)
        modules[f"decoder_{i}_expand"] = nn.Sequential(layer.fc1, nn.ReLU())
        modules[f"decoder_{i}_project"] = ProjectionResidualNorm(layer.fc2, layer.norm3)
    for module in modules.values():
        module.eval()
    samples = {name: [] for name in modules}
    def apply(name, *values):
        samples[name].append(tuple(x.detach().clone() for x in values))
        return modules[name](*values)
    with torch.no_grad():
        for text in CALIBRATION:
            trace = translator.translate(text, collect=True)
            hidden, mask = trace["encoder_inputs"]
            for i in range(6):
                hidden = apply(f"encoder_{i}_attention", hidden, mask)
                activated = apply(f"encoder_{i}_expand", hidden)
                hidden = apply(f"encoder_{i}_project", hidden, activated)
            traces = trace["decoder_inputs"]
            for t in sorted({0, min(1, len(traces)-1), len(traces)//2, len(traces)-1}):
                hidden, encoded, mask, last = traces[t]
                for i in range(2):
                    hidden = apply(f"decoder_{i}_self", hidden)
                    hidden = apply(f"decoder_{i}_cross", hidden, encoded, mask)
                    activated = apply(f"decoder_{i}_expand", hidden)
                    hidden = apply(f"decoder_{i}_project", hidden, activated)
    quantized = {}
    for name in (["encoder_0_project"] if args.first_probe else modules):
        quantized[name] = full_export.quantize(name, modules[name], samples[name])
    with torch.no_grad():
        hidden, mask = translator.encode_inputs("Please close the door")
        for i in range(6):
            hidden = modules[f"encoder_{i}_attention"](hidden, mask)
            activated = modules[f"encoder_{i}_expand"](hidden)
            name = f"encoder_{i}_project"
            if name in quantized:
                np.savez(OUTPUT / (name+"_fixture.npz"), input_0=hidden.numpy(),
                         input_1=activated.numpy(), output=quantized[name](hidden, activated).numpy())
            hidden = modules[name](hidden, activated)
    if args.first_probe:
        full_export.compile_model("encoder_0_project")
        return
    for i in range(4):
        name = f"head_{i}"
        source = ARTIFACTS / "staging" / (name+"_a16w8.pt2")
        shutil.copyfile(source, OUTPUT / source.name)
        quantized[name] = torch.export.load(source).module()
    split = SplitTranslator(quantized)
    results = []
    with torch.no_grad():
        for text in EVALUATION:
            baseline = translator.translate(text)
            actual = translator.translate(text, split.encoder, split.decoder)
            result = {"input": text, "float": baseline["text"], "split": actual["text"],
                      "exact_match": baseline["tokens"] == actual["tokens"],
                      "float_tokens": baseline["tokens"], "split_tokens": actual["tokens"],
                      "finished": actual["finished"]}
            results.append(result)
            print("EVALUATION", json.dumps(result), flush=True)
    report = {"denominator": len(results), "exact_matches": sum(r["exact_match"] for r in results),
              "finished": sum(r["finished"] for r in results), "results": results,
              "samples_sha256": hashlib.sha256((ARTIFACTS / "samples.json").read_bytes()).hexdigest()}
    (OUTPUT / "evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
    if not args.quantize_only:
        for name in quantized:
            full_export.compile_model(name)


if __name__ == "__main__":
    main()
