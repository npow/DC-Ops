#!/usr/bin/env python3
"""Synthetic HE OCR smoke test; never claims to test the live camera pipeline."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np
from PIL import Image

from direct_upload import DirectConnection
from load_assets import ARTIFACTS, NM, ROOT, load_symbols

MAILBOX = 0x02680000
HE_STATUS = 0x58800150
HE_ERROR = 0x5880014C
INPUT = 0xA2A00000
OUTPUT = 0xA3180000
INPUT_BYTES = 3 * 48 * 320 * 4
OUTPUT_BYTES = 40 * 97 * 4


def prepare(image_path):
    image = Image.open(image_path).convert("RGB")
    width = min(320, math.ceil(48 * image.width / image.height))
    # Matches ocr.recognition_input: PIL resize, BGR, normalized zero padding.
    pixels = np.asarray(image.resize((width, 48)))[:, :, ::-1].copy()
    tensor = np.zeros((1, 3, 48, 320), dtype="<f4")
    tensor[0, :, :, :width] = pixels.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
    return tensor, width


def decode(logits, characters, width):
    logits = logits[:math.ceil(40 * width / 320)]
    if not np.isfinite(logits).all():
        raise RuntimeError("HE output contains nonfinite logits")
    ids = logits.argmax(axis=-1)
    text = "".join(characters[int(token)] for i, token in enumerate(ids)
                   if token and (i == 0 or token != ids[i-1])).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ARTIFACTS / "ocr_calibration/word_00.png")
    parser.add_argument("--elf", type=Path, default=ARTIFACTS / "flashed_live.elf")
    parser.add_argument("--speed", type=int, default=15000)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Prepare fixture/report without loading J-Link or touching hardware")
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("Timeout must be positive")
    directory = ARTIFACTS / "he_smoke"
    directory.mkdir(exist_ok=True)
    tensor, width = prepare(args.image)
    raw = tensor.tobytes()
    assert len(raw) == INPUT_BYTES
    (directory / "input_f32.bin").write_bytes(raw)
    dictionary = (ARTIFACTS / "ocr-source/pytorchocr/utils/en_dict.txt").read_text().splitlines()
    characters = [""] + dictionary + [" "]
    assert len(characters) == 97
    reference = None
    for row in json.loads((ARTIFACTS / "ocr_calibration_manifest.json").read_text()):
        if args.image.resolve() == (ARTIFACTS / "ocr_calibration" / row["file"]).resolve():
            if hashlib.sha256(args.image.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError("Synthetic calibration image hash changed")
            reference = row["text"]
    report = {"scope": "synthetic printed-image HE recognizer smoke test; not live camera or translation",
              "image": str(args.image.resolve()), "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
              "input_sha256": hashlib.sha256(raw).hexdigest(), "reference": reference,
              "valid_width": width, "input_bytes": INPUT_BYTES, "output_bytes": OUTPUT_BYTES,
              "dry_run": args.dry_run}
    if args.dry_run:
        (directory / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    symbols = load_symbols(args.elf, NM)
    required = ("live_benchmark_enabled", "live_benchmark_command", "live_benchmark_status", "ocr_state")
    if any(name not in symbols for name in required):
        raise RuntimeError("Flashed ELF lacks required idle/paused observability flags")
    connection = DirectConnection()
    try:
        connection.connect(args)
        link = connection.link

        def words(address, count=1):
            values = link.memory_read32(address, count)
            if len(values) != count:
                raise RuntimeError(f"Short read at 0x{address:08x}")
            return [int(value) for value in values]

        def write_word(address, value):
            if link.memory_write32(address, [value]) != 4:
                raise RuntimeError(f"Word write failed at 0x{address:08x}")

        paused = words(symbols["live_benchmark_enabled"])[0]
        benchmark_command = words(symbols["live_benchmark_command"])[0]
        benchmark_state = words(symbols["live_benchmark_status"])[0]
        ocr_state = words(symbols["ocr_state"])[0]
        if paused != 1 or benchmark_command != 0 or benchmark_state in (1, 2, 3, 4):
            raise RuntimeError("HP is not safely paused: set live_benchmark_enabled=1, leave command=0, and wait for active work to finish; script does not change these flags")
        if ocr_state in (3, 4, 6, 7, 8):
            raise RuntimeError(f"OCR still executing (state={ocr_state}); wait for its return before HE smoke test")
        he_status, he_error = words(HE_STATUS)[0], words(HE_ERROR)[0]
        if he_status != 2:
            raise RuntimeError(f"HE worker is not ready: status={he_status}, error=0x{he_error:08x}")
        request_sequence = words(MAILBOX + 8)[0]
        response_sequence = words(MAILBOX + 128)[0]
        if request_sequence and request_sequence != response_sequence:
            raise RuntimeError("HE has an outstanding mailbox request; input will not be overwritten")
        sequence = (max(request_sequence, response_sequence) + 1) & 0xffffffff
        sequence = sequence or 1
        connection.write_bytes(INPUT, raw, time.monotonic() + args.timeout)
        # HP stays paused; HE invalidates input itself and cleans output before ack.
        request = struct.pack("<15I68x", 0x45384F43, 1, 0, 1, 1,
                              INPUT, INPUT_BYTES, 0, 0, 0, 0, 0, 0, OUTPUT, OUTPUT_BYTES)
        assert len(request) == 128
        connection.write_bytes(MAILBOX, request, time.monotonic() + args.timeout)
        started = time.monotonic()
        write_word(MAILBOX + 8, sequence)
        while True:
            response_raw = struct.pack("<32I", *words(MAILBOX + 128, 32))
            response = struct.unpack("<IiIIQIII92x", response_raw)
            if response[0] == sequence:
                break
            if time.monotonic() - started > args.timeout:
                raise RuntimeError("HE request timed out; its input remains owned by HE. Keep HP paused and do not overwrite input until the matching completion arrives")
            time.sleep(.1)
        elapsed = time.monotonic() - started
        if response[1] or response[2] != OUTPUT_BYTES:
            raise RuntimeError(f"HE completion error={response[1]}, output_bytes={response[2]}")
        output_raw = struct.pack(f"<{OUTPUT_BYTES//4}I", *words(OUTPUT, OUTPUT_BYTES//4))
        (directory / "logits_f32.bin").write_bytes(output_raw)
        logits = np.frombuffer(output_raw, dtype="<f4").reshape(40, 97)
        text = decode(logits, characters, width)
        report.update({"sequence": sequence, "recognized": text, "exact_match": text == reference if reference is not None else None,
                       "host_wait_seconds": elapsed, "worker_elapsed_us": response[3],
                       "npu_cycles": response[4], "npu_active": response[5], "mac_active": response[6], "npu_idle": response[7],
                       "output_sha256": hashlib.sha256(output_raw).hexdigest(), "hp_left_paused": True})
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
