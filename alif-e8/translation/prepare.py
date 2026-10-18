"""Download a pinned checkpoint and verify the fixed-shape translation adapter."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

from translator import Translator

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent / "artifacts/translator"
CHECKPOINT = ARTIFACTS / "checkpoint"
REPO = "Helsinki-NLP/opus-mt_tiny_eng-spa"
REVISION = "691fd47381a57d0de256a0e0b201bd83d1f48ba0"
CALIBRATION = [
    "Emergency exit", "Keep out", "No smoking", "Push to open", "Pull to open",
    "Please wash your hands", "Keep this door closed", "Entrance", "Danger high voltage",
    "Do not touch", "Wet floor", "Staff only", "Turn left", "Turn right",
    "Drinking water", "Out of service", "Mind the step", "Parking prohibited",
    "Open from nine to five", "Thank you for your visit", "Keep refrigerated",
    "First aid", "Please wait here", "Fire extinguisher",
]
EVALUATION = [
    "Welcome", "Do not enter", "Please close the door", "The store is open",
    "Where is the train station?", "Caution hot surface", "Press the green button",
    "This machine is out of order",
]


def load_model():
    model = MarianMTModel.from_pretrained(CHECKPOINT, local_files_only=True,
                                         attn_implementation="eager").float().eval()
    tokenizer = MarianTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    return model, tokenizer


def main():
    torch.set_num_threads(4)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    files = ["model.safetensors", "config.json", "generation_config.json", "source.spm",
             "target.spm", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json"]

    def download(name):
        path = CHECKPOINT / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", temporary)
            temporary.rename(path)
        return {"file": name, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download, files))
    (ARTIFACTS / "checkpoint_manifest.json").write_text(json.dumps(
        {"repository": REPO, "revision": REVISION, "files": manifest}, indent=2)+"\n")
    samples = {"calibration": CALIBRATION, "evaluation": EVALUATION}
    raw = json.dumps(samples, indent=2)+"\n"
    (ARTIFACTS / "samples.json").write_text(raw)
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    print("Model parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    results = []
    with torch.no_grad():
        for text in EVALUATION:
            inputs = tokenizer(text, return_tensors="pt")
            baseline_ids = model.generate(**inputs, max_new_tokens=31, num_beams=1, do_sample=False)
            baseline = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)
            actual = translator.translate(text)
            row = {"input": text, "huggingface": baseline, "fixed_shape": actual["text"],
                   "exact_match": actual["text"] == baseline, "finished": actual["finished"]}
            results.append(row)
            print(json.dumps(row), flush=True)
    report = {"denominator": len(results), "exact_matches": sum(r["exact_match"] for r in results),
              "samples_sha256": hashlib.sha256(raw.encode()).hexdigest(), "sequence_length": 32,
              "results": results}
    (ARTIFACTS / "baseline.json").write_text(json.dumps(report, indent=2)+"\n")
    assert report["exact_matches"] == len(results), report


if __name__ == "__main__":
    main()
