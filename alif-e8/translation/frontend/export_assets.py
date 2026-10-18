"""Export immutable MCU frontend assets and frozen host parity fixtures."""
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch
from sentencepiece import sentencepiece_model_pb2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import ARTIFACTS, CALIBRATION, CHECKPOINT, EVALUATION, REVISION, load_model
from translator import Translator

OUT = ARTIFACTS / "frontend"


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString((CHECKPOINT / "source.spm").read_bytes())
    nodes = [{"children": {}, "token": 0xffffffff, "score": 0.0}]
    for piece in proto.pieces:
        if piece.type != 1:
            continue
        node = 0
        for byte in piece.piece.encode():
            children = nodes[node]["children"]
            if byte not in children:
                children[byte] = len(nodes)
                nodes.append({"children": {}, "token": 0xffffffff, "score": 0.0})
            node = children[byte]
        nodes[node]["token"] = tokenizer.encoder.get(piece.piece, 1)
        nodes[node]["score"] = piece.score
    edges = []
    node_bytes = bytearray()
    for node in nodes:
        node_bytes.extend(struct.pack("<IHHIf", len(edges), len(node["children"]), 0, node["token"], node["score"]))
        edges.extend((child, byte) for byte, child in sorted(node["children"].items()))
    strings = bytearray()
    offsets = []
    for token in range(32001):
        offsets.append(len(strings))
        strings.extend(tokenizer.decoder[token].encode())
    offsets.append(len(strings))
    unknown_score = min(p.score for p in proto.pieces if p.type == 1) - 10.0
    lexicon = struct.pack("<8sIIIIfI", b"DCOPSTK1", 1, len(nodes), len(edges), len(strings), unknown_score, 0)
    lexicon += node_bytes + b"".join(struct.pack("<IB3x", child, byte) for child, byte in edges)
    lexicon += np.asarray(offsets, dtype="<u4").tobytes() + strings
    (OUT / "lexicon.bin").write_bytes(lexicon)
    (OUT / "embeddings_fp16.bin").write_bytes(model.model.shared.weight.detach().numpy().astype("<f2").tobytes())
    positions = model.model.encoder.embed_positions(torch.Size((1, 32))).detach().numpy().astype("<f4")
    assert torch.equal(model.model.encoder.embed_positions(torch.Size((1,32))), model.model.decoder.embed_positions(torch.Size((1,32))))
    (OUT / "positions_fp32.bin").write_bytes(positions.tobytes())
    extra = ["", "  A\tB\nC  ", "café naïve", "¿Dónde está el café?", "niño, acción y corazón.", "AA² º ¹", "Unknown §§ characters", "No\u00a0smoking"]
    samples = CALIBRATION + EVALUATION + extra
    fixture = bytearray(struct.pack("<I", len(samples)))
    rows = []
    with torch.no_grad():
        for text in samples:
            token_ids = tokenizer(text)["input_ids"]
            padded = token_ids + [32000] * (32 - len(token_ids))
            encoder_inputs, mask = translator.encode_inputs(text)
            encoded_text = text.encode()
            fixture.extend(struct.pack("<I", len(encoded_text)) + encoded_text)
            fixture.extend(struct.pack("<I", len(token_ids)) + np.asarray(padded, dtype="<u4").tobytes())
            fixture.extend(encoder_inputs.numpy().astype("<f4").tobytes() + mask.numpy().astype("<f4").tobytes())
            rows.append({"text": text, "tokens": token_ids})
    decodes = []
    for text in ["¿Dónde está el café?", "niño, acción y corazón.", "¡Bienvenidos!", "Por favor, cierre la puerta."]:
        ids = [tokenizer.encoder.get(piece, 1) for piece in tokenizer.spm_target.encode(text, out_type=str)]
        decodes.append({"ids": ids, "expected": tokenizer.decode(ids, skip_special_tokens=True)})
    with torch.no_grad():
        batch = tokenizer(CALIBRATION + EVALUATION, return_tensors="pt", padding=True)
        generated = model.generate(**batch, max_new_tokens=31, num_beams=1, do_sample=False)
    for source, ids in zip(CALIBRATION + EVALUATION, generated.tolist()):
        decodes.append({"source": source, "ids": ids, "expected": tokenizer.decode(ids, skip_special_tokens=True)})
    fixture.extend(struct.pack("<I", len(decodes)))
    for row in decodes:
        encoded = row["expected"].encode()
        fixture.extend(struct.pack("<I", len(row["ids"])) + np.asarray(row["ids"], dtype="<u4").tobytes())
        fixture.extend(struct.pack("<I", len(encoded)) + encoded)
    prefix = [32000] + decodes[0]["ids"]
    embeddings, _, _, last = translator.decoder_inputs(prefix, torch.zeros(1,32,256), torch.zeros(1,1,1,32))
    fixture.extend(struct.pack("<I", len(prefix)) + np.asarray(prefix, dtype="<u4").tobytes())
    fixture.extend(embeddings.detach().numpy().astype("<f4").tobytes() + last.numpy().astype("<f4").tobytes())
    (OUT / "fixtures.bin").write_bytes(fixture)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.glob("*.bin")}
    manifest = {"revision": REVISION, "samples_sha256": hashlib.sha256((ARTIFACTS / "samples.json").read_bytes()).hexdigest(), "assets": files,
                "tokenizer_nodes": len(nodes), "tokenizer_edges": len(edges), "source_samples": rows, "decode_samples": decodes}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"nodes": len(nodes), "edges": len(edges), "assets": files}, indent=2))


if __name__ == "__main__":
    main()
