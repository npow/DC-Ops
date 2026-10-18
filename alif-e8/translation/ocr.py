"""Load pinned PP-OCR text detection and English recognition checkpoints."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from prepare import ARTIFACTS

SOURCE = ARTIFACTS / "ocr-source"
CHECKPOINT = ARTIFACTS / "ocr-checkpoint"
sys.path.insert(0, str(SOURCE))
from pytorchocr.modeling.backbones.det_mobilenet_v3 import MobileNetV3
from pytorchocr.modeling.backbones.rec_lcnetv3 import PPLCNetV3
from pytorchocr.modeling.necks.db_fpn import RSEFPN
from pytorchocr.modeling.necks.rnn import SequenceEncoder
from pytorchocr.modeling.heads.det_db_head import DBHead


class TextDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV3(scale=0.5, model_name="large", disable_se=True)
        self.neck = RSEFPN(in_channels=self.backbone.out_channels, out_channels=96, shortcut=True)
        self.head = DBHead(in_channels=96).binarize

    def forward(self, image):
        return self.head(self.neck(self.backbone(image)))


class TextRecognizer(nn.Module):
    def __init__(self, characters):
        super().__init__()
        self.backbone = PPLCNetV3(scale=0.95)
        self.encoder = SequenceEncoder(self.backbone.out_channels, encoder_type="svtr",
            dims=120, depth=2, hidden_dims=120, kernel_size=[1,3], use_guide=True)
        self.head = nn.Linear(self.encoder.out_channels, characters)

    def forward(self, image):
        return self.head(self.encoder(self.backbone(image)))


def load_ocr():
    detector = TextDetector()
    weights = torch.load(CHECKPOINT / "en_ptocr_v3_det_infer.pth", map_location="cpu", weights_only=True)
    det_state = {k.replace("head.binarize.", "head."): v
                 for k, v in weights.items() if not k.startswith("head.thresh.")}
    detector.load_state_dict(det_state)
    dictionary = (SOURCE / "pytorchocr/utils/en_dict.txt").read_text().splitlines()
    characters = [""] + dictionary + [" "]
    recognizer = TextRecognizer(len(characters))
    weights = torch.load(CHECKPOINT / "en_ptocr_v4_rec_infer.pth", map_location="cpu", weights_only=True)
    rec_state = {k.replace("head.ctc_encoder.", "encoder.").replace("head.ctc_head.fc.", "head."): v
                 for k, v in weights.items() if k.startswith(("backbone.", "head.ctc_encoder.", "head.ctc_head.fc."))}
    recognizer.load_state_dict(rec_state)
    return detector.eval(), recognizer.eval(), characters


def make_word(text, font_size=32):
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
    box = font.getbbox(text)
    image = Image.new("RGB", (box[2]-box[0]+16, box[3]-box[1]+16), "white")
    ImageDraw.Draw(image).text((8-box[0], 8-box[1]), text, font=font, fill="black")
    return image


def recognition_input(image):
    width = min(320, math.ceil(48*image.width/image.height))
    pixels = np.asarray(image.convert("RGB").resize((width, 48)))[:, :, ::-1].copy()
    tensor = torch.zeros((1, 3, 48, 320))
    tensor[0, :, :, :width] = torch.from_numpy(pixels).permute(2, 0, 1).float()/127.5-1
    return tensor


def decode(logits, characters, valid_width=320):
    logits = logits[:, :math.ceil(logits.shape[1]*valid_width/320)]
    tokens = logits.argmax(-1).flatten().tolist()
    return "".join(characters[t] for i, t in enumerate(tokens) if t and (i == 0 or t != tokens[i-1]))


def main():
    torch.set_num_threads(4)
    detector, recognizer, chars = load_ocr()
    print("PARAMETERS", sum(p.numel() for p in detector.parameters()), sum(p.numel() for p in recognizer.parameters()), flush=True)
    tests = ["Welcome", "Do not enter", "Please close the door", "Emergency exit", "Open", "Keep out"]
    results = []
    with torch.no_grad():
        for text in tests:
            image = make_word(text)
            actual = decode(recognizer(recognition_input(image)), chars)
            row = {"reference": text, "recognized": actual, "exact_match": text == actual}
            results.append(row)
            print(json.dumps(row), flush=True)
        scene = Image.new("RGB", (288, 480), "white")
        scene.paste(make_word("Welcome", 32), (24, 100))
        scene.paste(make_word("Keep out", 32), (24, 240))
        pixels = torch.from_numpy(np.asarray(scene)[:, :, ::-1].copy()).permute(2, 0, 1).float()/255
        tensor = ((pixels-torch.tensor([.485,.456,.406])[:,None,None]) / torch.tensor([.229,.224,.225])[:,None,None]).unsqueeze(0)
        maps = detector(tensor)
        print("DETECTOR", maps.shape, maps.min().item(), maps.max().item(), flush=True)
        scene.save(ARTIFACTS / "ocr_scene.png")
        Image.fromarray((maps[0,0].numpy()*255).astype("uint8")).save(ARTIFACTS / "ocr_detection.png")
    manifest = [{"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in CHECKPOINT.glob("*.pth")]
    report = {"denominator": len(results), "exact_matches": sum(r["exact_match"] for r in results),
              "results": results, "checkpoint_manifest": manifest, "characters": chars,
              "scope": "synthetic printed text smoke test; not a real-camera accuracy evaluation"}
    (ARTIFACTS / "ocr_baseline.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
