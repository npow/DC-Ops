"""Evaluate frozen target-vocabulary shortlists without changing source tokenization."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import ARTIFACTS, CHECKPOINT, EVALUATION, REVISION, load_model
from translator import Translator


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class ShortlistedDecoder(nn.Module):
    def __init__(self, decoder, ids):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("ids", torch.tensor(ids, dtype=torch.long))
        self.register_buffer("weight", decoder.head.weight.detach()[self.ids].clone())
        self.register_buffer("bias", decoder.bias[:, self.ids].clone())

    def forward(self, hidden, encoded, mask, last):
        for layer in self.decoder.layers:
            hidden = layer(hidden, encoded, self.decoder.causal_mask, mask)
        selected = torch.nn.functional.linear((hidden * last).sum(dim=1), self.weight) + self.bias
        logits = selected.new_full((1, self.decoder.head.out_features), -float("inf"))
        logits[:, self.ids] = selected
        return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "shortlist_evaluation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checkpoint_manifest = json.loads((ARTIFACTS / "checkpoint_manifest.json").read_bytes())
    assert checkpoint_manifest["revision"] == REVISION
    for entry in checkpoint_manifest["files"]:
        raw = (CHECKPOINT / entry["file"]).read_bytes()
        assert len(raw) == entry["bytes"] and digest(raw) == entry["sha256"], entry["file"]
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    vocabulary = model.lm_head.out_features
    spm = tokenizer.spm_target
    ranked = []
    characters = set()
    missing_pieces = []
    for index in range(spm.get_piece_size()):
        piece = spm.id_to_piece(index)
        token = tokenizer.encoder.get(piece)
        if token is None or token >= vocabulary:
            missing_pieces.append(piece)
            continue
        ranked.append((float(spm.get_score(index)), token))
        if len(piece.removeprefix("▁")) == 1:
            characters.add(token)
    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    specials = {token for token in tokenizer.all_special_ids if 0 <= token < vocabulary}
    mandatory = characters | specials
    assert model.config.eos_token_id in mandatory
    selections = {}
    for size in (2048, 4096, 8192):
        if len(mandatory) > size:
            raise ValueError(f"Mandatory character/special tokens ({len(mandatory)}) exceed {size}")
        selected = set(mandatory)
        for _, token in ranked:
            if len(selected) == size:
                break
            selected.add(token)
        assert len(selected) == size
        ids = sorted(selected)
        raw = (json.dumps(ids, separators=(",", ":")) + "\n").encode()
        (args.output / f"ids_{size}.json").write_bytes(raw)
        selections[size] = {"ids": ids, "selection_sha256": digest(raw)}
    samples_raw = (ARTIFACTS / "samples.json").read_bytes()
    assert json.loads(samples_raw)["evaluation"] == EVALUATION
    diagnostics = ["T", "Hello", "STOP"]
    report = {
        "checkpoint_revision": REVISION,
        "checkpoint_manifest_sha256": digest((ARTIFACTS / "checkpoint_manifest.json").read_bytes()),
        "target_spm_sha256": digest((CHECKPOINT / "target.spm").read_bytes()),
        "samples_sha256": digest(samples_raw),
        "evaluation_inputs_sha256": digest(json.dumps(EVALUATION, ensure_ascii=False).encode()),
        "selection_policy": "Descending target SentencePiece unigram score, original token-ID tie break; all single Unicode character pieces (after optional leading boundary marker) and valid special IDs mandatory. Selection occurs before evaluation; no evaluation-derived tokens.",
        "character_piece_count": len(characters), "mandatory_count": len(mandatory),
        "missing_target_pieces": missing_pieces,
        "invalid_special_ids_ignored": [i for i in tokenizer.all_special_ids if i >= vocabulary],
        "original_parameters": sum(p.numel() for p in model.parameters()),
        "original_head_rows": vocabulary,
        "checkpoint_files_verified": len(checkpoint_manifest["files"]),
        "full_precision_baseline": [], "shortlists": {},
        "limitations": "Restricted greedy output vocabulary changes predictions. Character availability does not guarantee spell-out behavior; longer outputs can hit the unchanged 31-token generation limit. Eight frozen examples are a smoke test, not a general quality benchmark. INT8 size estimates exclude PTE headers, command streams and compression.",
    }
    baselines = {}
    with torch.inference_mode():
        for text in EVALUATION + diagnostics:
            baseline = translator.translate(text)
            baselines[text] = {key: baseline[key] for key in ("text", "tokens", "finished")}
            report["full_precision_baseline"].append({"input": text, **baselines[text]})
        for size, selection in selections.items():
            decoder = ShortlistedDecoder(translator.decoder, selection["ids"]).eval()
            rows = []
            for text in EVALUATION + diagnostics:
                actual = translator.translate(text, decoder=decoder)
                baseline = baselines[text]
                row = {"input": text, "diagnostic": text in diagnostics,
                       "baseline": baseline, "shortlist": {key: actual[key] for key in ("text", "tokens", "finished")},
                       "exact_tokens": actual["tokens"] == baseline["tokens"],
                       "exact_text": actual["text"] == baseline["text"],
                       "token_length": len(actual["tokens"]), "baseline_token_length": len(baseline["tokens"]),
                       "extra_tokens": len(actual["tokens"]) - len(baseline["tokens"]),
                       "baseline_tokens_not_selected": sorted(set(baseline["tokens"]) - set(selection["ids"]))}
                rows.append(row)
                print(json.dumps({"size": size, **row}, ensure_ascii=False), flush=True)
            evaluation = [row for row in rows if not row["diagnostic"]]
            report["shortlists"][str(size)] = {
                "selection_sha256": selection["selection_sha256"], "selected_rows": size,
                "int8_head_weight_bytes": size * 256,
                "head_weight_reduction_fraction": 1 - size / vocabulary,
                "denominator": len(evaluation),
                "exact_token_matches": sum(row["exact_tokens"] for row in evaluation),
                "exact_text_matches": sum(row["exact_text"] for row in evaluation),
                "finished": sum(row["shortlist"]["finished"] for row in evaluation),
                "longer_outputs": sum(row["extra_tokens"] > 0 for row in evaluation),
                "diagnostic_denominator": len(diagnostics), "results": rows}
            (args.output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("REPORT", args.output / "evaluation.json", flush=True)


if __name__ == "__main__":
    main()
