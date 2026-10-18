#!/usr/bin/env python3
"""Verify every asset block on the E8 and rewrite only CRC-mismatching blocks."""
import argparse
import contextlib
import hashlib
import io
import json
import math
import struct
import sys
import time
import zlib
from pathlib import Path

from direct_upload import DirectConnection
from load_assets import ARTIFACTS, ASSETS, NM, load_symbols

BLOCK_BYTES = 256
BAD_CAPACITY = 262144
FIELDS = ("command", "address", "bytes", "block_size", "status", "error", "bad_count",
          "stored_count", "blocks_done", "block_count", "elapsed_ms")


def read_word(link, address):
    values = link.memory_read32(address, 1)
    if len(values) != 1:
        raise RuntimeError(f"Short word read at0x{address:08x}")
    return values[0]


def write_word(link, address, value):
    if link.memory_write32(address, [value & 0xffffffff]) != 4:
        raise RuntimeError(f"Short word write at0x{address:08x}")


def expected_crc(raw):
    return [zlib.crc32(raw[offset:offset+BLOCK_BYTES]) & 0xffffffff
            for offset in range(0, len(raw), BLOCK_BYTES)]


def scan(connection, symbols, address, raw, table, timeout, directory, label):
    link = connection.link
    field = lambda name: symbols["live_integrity_"+name]
    if read_word(link, field("command")) or read_word(link, field("status")) == 1:
        raise RuntimeError("Integrity service is already busy; staging memory was not overwritten")
    staging = symbols["live_model_staging"]
    if len(table) > 1024*1024:
        raise RuntimeError("Expected CRC table exceeds1MiB staging prefix")
    expected_pointer = (read_word(link, symbols["live_integrity_expected_address"])
                        if "live_integrity_expected_address" in symbols else staging)
    bad_pointer = (read_word(link, symbols["live_integrity_bad_indices_address"])
                   if "live_integrity_bad_indices_address" in symbols else staging+1024*1024)
    if expected_pointer not in (0, staging) or bad_pointer not in (0, staging+1024*1024):
        raise RuntimeError("Integrity service scratch addresses differ from the agreed SRAM layout")
    started = time.monotonic()
    connection.write_bytes(staging, table, started+timeout)
    for name, value in (("address", address), ("bytes", len(raw)), ("block_size", BLOCK_BYTES),
                        ("status", 0), ("blocks_done", 0), ("error", 0), ("command", 1)):
        write_word(link, field(name), value)
    previous_progress = -1
    while True:
        state = {name: read_word(link, field(name)) for name in FIELDS
                 if name not in ("address", "bytes", "block_size")}
        if state["blocks_done"] != previous_progress:
            print(f"{label}: scanned{state['blocks_done']}/{math.ceil(len(raw)/BLOCK_BYTES)} blocks, "
                  f"bad{state['bad_count']}", flush=True)
            previous_progress = state["blocks_done"]
        if state["status"] == 0xe0:
            signed = struct.unpack("<i", struct.pack("<I", state["error"]))[0]
            raise RuntimeError(f"Integrity service failed, error={signed}, state={state}")
        if state["command"] == 0 and state["status"] == 2:
            break
        if time.monotonic()-started >= timeout:
            raise RuntimeError(f"Integrity scan timed out after{timeout}s; no further staging writes issued")
        time.sleep(0.05)
    expected_blocks = math.ceil(len(raw)/BLOCK_BYTES)
    if state["error"] or state["block_count"] != expected_blocks or state["blocks_done"] != expected_blocks:
        raise RuntimeError(f"Integrity scan did not cover the complete asset: {state}")
    if state["bad_count"] > BAD_CAPACITY or state["stored_count"] != state["bad_count"]:
        raise RuntimeError(f"Mismatch list overflow or incomplete list: {state}")
    indices = list(link.memory_read32(staging+1024*1024, state["stored_count"])) if state["stored_count"] else []
    if len(indices) != state["bad_count"] or len(set(indices)) != len(indices) or any(i >= expected_blocks for i in indices):
        raise RuntimeError("Bad-block list is truncated, duplicated, or outside the asset")
    result = {**state, "bad_indices": indices, "host_elapsed_seconds": round(time.monotonic()-started, 3)}
    (directory / (label+".json")).write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, default=ARTIFACTS / "flashed_live.elf")
    parser.add_argument("--assets", type=Path, default=ARTIFACTS)
    parser.add_argument("--nm", type=Path, default=NM)
    parser.add_argument("--speed", type=int, default=15000)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-passes", type=int, default=3, help="Maximum complete scans per asset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_passes < 1:
        raise ValueError("At least one scan pass is required")
    assets = []
    for relative, address, size in ASSETS:
        path = args.assets / relative
        raw = path.read_bytes()
        if len(raw) != size:
            raise RuntimeError(f"Wrong asset size: {path}")
        crc = expected_crc(raw)
        table = struct.pack("<"+"I"*len(crc), *crc)
        assets.append({"file": relative, "address": address, "bytes": size,
                       "source_sha256": hashlib.sha256(raw).hexdigest(),
                       "blocks": len(crc), "crc_table_bytes": len(table), "raw": raw, "table": table})
    summary = {"block_bytes": BLOCK_BYTES, "asset_count": len(assets),
               "payload_bytes": sum(a["bytes"] for a in assets),
               "total_blocks": sum(a["blocks"] for a in assets),
               "crc_upload_bytes": sum(a["crc_table_bytes"] for a in assets),
               "assets": [{k: v for k, v in a.items() if k not in ("raw", "table")} for a in assets],
               "verification": "E8 CPU CRC32/zlib comparison of every256-byte block including partial tails; not cryptographic full-read SHA verification"}
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; no hardware access.")
        return
    symbols = load_symbols(args.elf, args.nm)
    for name in ("live_model_staging", *("live_integrity_"+name for name in FIELDS)):
        if name not in symbols:
            raise RuntimeError(f"ELF is missing{name}; integrity helper must be flashed first")
    directory = ARTIFACTS / "asset_repair" / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
    connection = DirectConnection()
    report = {"assets": [], "verification": summary["verification"], "complete": False}
    try:
        connection.connect(args)
        link = connection.link
        if read_word(link, symbols["live_assets_ready"]) != 0 or read_word(link, symbols["live_pipeline_state"]) != 2:
            raise RuntimeError("Repair requires live_assets_ready0 and live_pipeline_state2; no assets were modified")
        if read_word(link, 0xe000ed08) != 0x80200000 or read_word(link, 0x80200004) != (symbols["Reset_Handler"] | 1):
            raise RuntimeError("Running vector table differs from the supplied live ELF")
        frames_before = read_word(link, symbols["live_frames"]) if "live_frames" in symbols else None
        for index, asset in enumerate(assets):
            asset_report = {k: v for k, v in asset.items() if k not in ("raw", "table")}
            asset_report.update(scans=[], repaired_blocks=0, repaired_bytes=0, verified_blocks=0)
            report["assets"].append(asset_report)
            previous_bad = None
            for attempt in range(args.max_passes):
                label = f"asset_{index}_pass_{attempt+1}"
                result = scan(connection, symbols, asset["address"], asset["raw"], asset["table"],
                              args.timeout, directory, label)
                asset_report["scans"].append(result)
                bad = result["bad_count"]
                if previous_bad is not None and bad > previous_bad:
                    raise RuntimeError(f"Bad-block count grew{previous_bad}→{bad} for{asset['file']}; stopping")
                if not bad:
                    asset_report["verified_blocks"] = asset["blocks"]
                    print(f"Verified ALL{asset['blocks']} blocks: {asset['file']}", flush=True)
                    break
                if attempt+1 == args.max_passes:
                    raise RuntimeError(f"{asset['file']} still has{bad} bad blocks after{args.max_passes} scans")
                print(f"Repairing{bad} blocks in{asset['file']}", flush=True)
                for repair_index, block in enumerate(result["bad_indices"]):
                    offset = block*BLOCK_BYTES
                    payload = asset["raw"][offset:offset+BLOCK_BYTES]
                    with contextlib.redirect_stdout(io.StringIO()):
                        connection.write_bytes(asset["address"]+offset, payload, time.monotonic()+args.timeout)
                    asset_report["repaired_blocks"] += 1
                    asset_report["repaired_bytes"] += len(payload)
                    if (repair_index+1)%128 == 0 or repair_index+1 == bad:
                        print(f"Repaired{repair_index+1}/{bad} blocks", flush=True)
                previous_bad = bad
            (directory / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        report["verified_blocks"] = sum(a["verified_blocks"] for a in report["assets"])
        report["total_blocks"] = summary["total_blocks"]
        report["repaired_bytes"] = sum(a["repaired_bytes"] for a in report["assets"])
        report["camera_frames_advanced"] = ((read_word(link, symbols["live_frames"])-frames_before) & 0xffffffff) if frames_before is not None else None
        report["complete"] = True
        print(json.dumps({"complete": True, "verified_blocks": report["verified_blocks"],
            "total_blocks": report["total_blocks"], "repaired_bytes": report["repaired_bytes"],
            "camera_frames_advanced": report["camera_frames_advanced"],
            "assets": [{"file": a["file"], "verified_blocks": a["verified_blocks"],
                        "repaired_blocks": a["repaired_blocks"], "scans": len(a["scans"])} for a in report["assets"]],
            "report": str(directory / "result.json"), "verification": summary["verification"]}, indent=2))
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        connection.close()
        (directory / "result.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Asset repair failed: {error}", file=sys.stderr)
        raise SystemExit(1)
