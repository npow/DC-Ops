#!/usr/bin/env python3
"""Upload mapped live-translation assets without halting or resetting the E8."""
import argparse
import hashlib
import json
import os
import re
import selectors
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/translator"
NM = Path("/Users/npow/.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin/arm-none-eabi-nm")
PIPELINE_DIRECTORY = "spatial_staging"
ASSETS = [
    ("frontend/embeddings_fp16.bin", 0xA0000000, 16384512),
    ("frontend/lexicon.bin", 0xA1000000, 2337119),
    ("frontend/positions_fp32.bin", 0xA1240000, 32768),
    (f"{PIPELINE_DIRECTORY}/pipeline.bin", 0xA1300000, 17056896),
    ("small_detector_128x224/text_detector_u55_256.pte", 0xA2400000, 667232),
    ("text_recognizer_ethos-u55-128.pte", 0xA2500000, 2198384),
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def quoted(path):
    value = str(path.resolve())
    if any(c in value for c in ('"', '\n', '\r')):
        raise ValueError(f"Unsupported J-Link path: {value!r}")
    return '"'+value+'"'


def load_symbols(elf, nm):
    if not elf.is_file():
        raise RuntimeError(f"Live ELF does not exist: {elf}")
    result = subprocess.run([str(nm), "-n", str(elf)], text=True, capture_output=True, check=True)
    symbols = {parts[2]: int(parts[0], 16) for line in result.stdout.splitlines()
               if len(parts := line.split()) == 3 and re.fullmatch(r"[0-9a-fA-F]+", parts[0])}
    for name in ("live_assets_ready", "live_pipeline_state", "Reset_Handler"):
        if name not in symbols:
            raise RuntimeError(f"{elf} has no {name}; supply the ELF actually flashed with --elf")
    return symbols


def jlink(args, directory, name, commands, timeout=None):
    script = "\n".join(["exec SetEnableFlashbank 0xA0000000=0",
                         "exec SetAllowFlashCache=0", *commands, "exit", ""])
    script_path = directory / (name+".jlink")
    script_path.write_text(script)
    command = [args.jlink, "-device", "AE822FA0E5597LS0_M55_HP", "-if", "SWD",
               "-speed", str(args.speed), "-autoconnect", "1"]
    timeout = args.timeout if timeout is None else timeout
    started = heartbeat = time.monotonic()
    pieces = []
    expired = False
    with script_path.open("rb") as source, (directory / (name+".log")).open("wb") as logfile:
        process = subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                now = time.monotonic()
                if now-started > timeout:
                    expired = True
                    process.kill()
                for key, _ in selector.select(timeout=1):
                    raw = os.read(key.fd, 65536)
                    if not raw:
                        selector.unregister(key.fileobj)
                        continue
                    pieces.append(raw)
                    logfile.write(raw)
                    logfile.flush()
                    sys.stdout.write(raw.decode(errors="replace"))
                    sys.stdout.flush()
                if now-heartbeat >= 15:
                    print(f"\n[J-Link {name}: {int(now-started)} seconds elapsed]", flush=True)
                    heartbeat = now
        returncode = process.wait()
    log = b"".join(pieces).decode(errors="replace")
    if expired:
        raise RuntimeError(f"J-Link {name} timed out after {timeout} seconds; inspect its streamed log")
    bad = re.search(r"\bERROR\b|cannot connect|could not (?:read|write|connect|find)|failed to|"
                    r"unable to|unknown command|syntax error|unspecified error|download failed", log, re.I)
    if returncode or bad:
        raise RuntimeError(f"J-Link {name} failed; see {directory / (name+'.log')}\n{log[-2000:]}")


def word(path):
    raw = path.read_bytes()
    if len(raw) != 4:
        raise RuntimeError(f"Expected 4-byte readback: {path}")
    return struct.unpack("<I", raw)[0]


def readback(directory, label, address, size):
    path = directory / (label+".bin")
    path.unlink(missing_ok=True)
    return path, f"savebin {quoted(path)} 0x{address:08x} 0x{size:x}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, default=ROOT / "out/translator-live/DevKit-E8/Debug/translator-live.elf")
    parser.add_argument("--assets", type=Path, default=ARTIFACTS)
    parser.add_argument("--nm", type=Path, default=NM)
    parser.add_argument("--jlink", default="JLinkExe")
    parser.add_argument("--speed", type=int, default=15000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--chunk-bytes", "--chunk-size", dest="chunk_bytes", type=lambda x: int(x, 0), help="Direct default256KiB; staging default1MiB")
    parser.add_argument("--chunk-timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=1, help="Retry a failed direct chunk this many times")
    parser.add_argument("--via-staging", action="store_true", help="Use firmware live_asset_service SRAM-to-PSRAM copy helper")
    parser.add_argument("--verify-only", action="store_true", help="Verify retained PSRAM assets without uploading or using staging helper")
    parser.add_argument("--resume", action="store_true", help="Skip chunks whose fresh PSRAM first/last128-byte readbacks match; not a full-chunk hash")
    parser.add_argument("--vector-base", type=lambda x: int(x, 0), default=0x80200000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-enable", action="store_true", help="Upload and verify assets, leaving live_assets_ready at zero")
    args = parser.parse_args()
    chunk_bytes = args.chunk_bytes or (1024*1024 if args.via_staging else 256*1024)
    if chunk_bytes <= 0 or (args.via_staging and chunk_bytes > 1024*1024):
        raise ValueError("Chunk size must be positive and staging chunks must be at most1MiB")
    assets = []
    previous_end = 0xA0000000
    frontend_manifest = json.loads((args.assets / "frontend/manifest.json").read_text())
    pipeline_manifest = json.loads((args.assets / PIPELINE_DIRECTORY / "pipeline_index.json").read_text())
    for relative, address, expected_size in ASSETS:
        path = args.assets / relative
        raw = path.read_bytes()
        if len(raw) != expected_size:
            raise RuntimeError(f"{path}: expected {expected_size} bytes, found {len(raw)}")
        if address < previous_end or address+len(raw) > 0xA4000000:
            raise RuntimeError(f"Asset map overlaps or exceeds PSRAM: {relative}")
        previous_end = address+len(raw)
        sha = digest(raw)
        expected_sha = (frontend_manifest["assets"][Path(relative).name]["sha256"]
                        if relative.startswith("frontend/") else
                        pipeline_manifest["sha256"] if relative == f"{PIPELINE_DIRECTORY}/pipeline.bin" else None)
        if expected_sha and sha != expected_sha:
            raise RuntimeError(f"Asset hash differs from its manifest: {path}")
        windows = {(0, min(128, len(raw))), (max(0, len(raw)-128), min(128, len(raw))),
                   ((len(raw)//2)//32*32, 128)}
        if relative == f"{PIPELINE_DIRECTORY}/pipeline.bin":
            for stage in pipeline_manifest["stages"]:
                start, count = stage["offset"], stage["bytes"]
                if digest(raw[start:start+count]) != stage["sha256"]:
                    raise RuntimeError(f"Packed stage hash mismatch: {stage['name']}")
                windows.update({(start, min(128, count)), (start+count-min(128, count), min(128, count))})
        assets.append({"file": str(path.resolve()), "address": address, "bytes": len(raw),
                       "sha256": sha, "windows": sorted(windows)})
    symbols = load_symbols(args.elf, args.nm) if args.elf.is_file() or not args.dry_run else None
    if symbols and args.via_staging and not args.verify_only:
        for name in ("live_model_staging", "live_asset_command", "live_asset_destination", "live_asset_bytes",
                     "live_asset_error", "live_asset_sequence", "live_asset_completed_sequence"):
            if name not in symbols:
                raise RuntimeError(f"ELF has no {name}; staging firmware helper is not linked")
    summary = {"elf": str(args.elf.resolve()), "speed_khz": args.speed,
               "total_bytes": sum(a["bytes"] for a in assets), "assets": assets,
               "ready_address": symbols["live_assets_ready"] if symbols else None,
               "dry_run": args.dry_run, "enable_after_upload": not args.no_enable,
               "chunk_bytes": chunk_bytes, "via_staging": args.via_staging,
               "verify_only": args.verify_only, "resume": args.resume}
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; no J-Link connection or target access.")
        return
    directory = ARTIFACTS / "live_asset_load" / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
    preflight, precommands = {}, []
    fields = {"ready": symbols["live_assets_ready"], "state": symbols["live_pipeline_state"],
              "vtor": 0xE000ED08, "reset_vector": args.vector_base+4}
    if "live_frames" in symbols:
        fields["frames"] = symbols["live_frames"]
    if args.via_staging and not args.verify_only:
        fields["asset_command"] = symbols["live_asset_command"]
    for name, address in fields.items():
        path, command = readback(directory, "before_"+name, address, 4)
        preflight[name] = path
        precommands.append(command)
    jlink(args, directory, "preflight", precommands)
    before = {key: word(path) for key, path in preflight.items()}
    if before["vtor"] != args.vector_base or before["reset_vector"] != (symbols["Reset_Handler"] | 1):
        raise RuntimeError("Running vector table does not match the live ELF; assets were not uploaded")
    if before["ready"] != 0 or before["state"] != 2:
        raise RuntimeError(f"Live firmware must be waiting for assets (ready=0,state=2), found {before}; no upload performed")
    if args.via_staging and not args.verify_only and before["asset_command"]:
        raise RuntimeError("Staging helper is busy; SRAM was not overwritten")
    commands, samples = [], []
    started = time.monotonic()
    uploaded = 0
    skipped = 0
    skipped_chunks = 0
    sequence = 0
    for index, asset in enumerate(assets):
        path = Path(asset["file"])
        raw = path.read_bytes()
        for offset in ([] if args.verify_only else range(0, len(raw), chunk_bytes)):
            chunk = raw[offset:offset+chunk_bytes]
            chunk_name = f"chunk_{index}_{offset:08x}"
            chunk_path = directory / (chunk_name+".bin")
            chunk_path.write_bytes(chunk)
            destination = asset["address"]+offset
            chunk_commands = []
            chunk_started = time.monotonic()
            def remaining_timeout():
                remaining = args.chunk_timeout-(time.monotonic()-chunk_started)
                if remaining <= 0:
                    raise RuntimeError(f"{chunk_name} exceeded {args.chunk_timeout} seconds")
                return remaining
            if args.resume:
                checks, reads = [], []
                for edge in sorted({0, max(0, len(chunk)-128)}):
                    target, command = readback(directory, chunk_name+f"_resume_{edge:08x}", destination+edge, min(128, len(chunk)))
                    reads.append(command)
                    checks.append((target, chunk[edge:edge+128]))
                jlink(args, directory, chunk_name+"_resume", reads, remaining_timeout())
                if all(target.read_bytes() == expected for target, expected in checks):
                    skipped += len(chunk)
                    skipped_chunks += 1
                    print(f"\nSkipped retained chunk: asset{index+1}/6 offset0x{offset:x}, "
                          f"{uploaded+skipped}/{summary['total_bytes']} bytes processed "
                          "(first/last128-byte samples match; full chunk not hashed)", flush=True)
                    continue
            if args.via_staging:
                if sequence == 0:
                    previous, command = readback(directory, "previous_sequence", symbols["live_asset_completed_sequence"], 4)
                    jlink(args, directory, "staging_sequence", [command], remaining_timeout())
                    sequence = word(previous)
                sequence = (sequence+1) & 0xffffffff or 1
                chunk_commands.append(f"loadfile {quoted(chunk_path)} 0x{symbols['live_model_staging']:08x} noreset")
                for field, value in (("destination", destination), ("bytes", len(chunk)), ("sequence", sequence), ("command", 1)):
                    chunk_commands.append(f"w4 0x{symbols['live_asset_'+field]:08x} 0x{value:x}")
                jlink(args, directory, chunk_name, chunk_commands, remaining_timeout())
                deadline = time.monotonic()+args.chunk_timeout
                poll = 0
                while True:
                    status, reads = {}, []
                    for field in ("command", "completed_sequence", "error"):
                        target, command = readback(directory, chunk_name+"_"+field, symbols["live_asset_"+field], 4)
                        status[field] = target
                        reads.append(command)
                    jlink(args, directory, chunk_name+f"_poll_{poll}", reads, remaining_timeout())
                    if word(status["command"]) == 0 and word(status["completed_sequence"]) == sequence:
                        if word(status["error"]):
                            raise RuntimeError(f"Staging helper failed on {chunk_name}, error={word(status['error'])}")
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Staging helper did not acknowledge {chunk_name}; SRAM was not overwritten")
                    poll += 1
            else:
                chunk_commands.append(f"loadfile {quoted(chunk_path)} 0x{destination:08x} noreset")
            checks = []
            reads = []
            for edge in sorted({0, max(0, len(chunk)-128)}):
                target, command = readback(directory, chunk_name+f"_edge_{edge:08x}", destination+edge, min(128, len(chunk)))
                reads.append(command)
                checks.append((target, chunk[edge:edge+128]))
            attempts = 1 if args.via_staging else args.retries+1
            for attempt in range(attempts):
                try:
                    jlink(args, directory, chunk_name+f"_verify_{attempt}",
                          ([] if args.via_staging else chunk_commands)+reads, remaining_timeout())
                    for target, expected in checks:
                        if target.read_bytes() != expected:
                            raise RuntimeError(f"Chunk readback differs: {target}")
                    break
                except RuntimeError:
                    if attempt+1 == attempts:
                        raise
                    print(f"Retrying {chunk_name}, attempt{attempt+2}/{attempts}", flush=True)
            uploaded += len(chunk)
            print(f"\nUploaded {uploaded} bytes, skipped{skipped}; "
                  f"{uploaded+skipped}/{summary['total_bytes']} bytes processed "
                  f"({uploaded/1048576:.2f}MiB), asset{index+1}/6 offset0x{offset:x}, "
                  f"elapsed{time.monotonic()-started:.1f}s", flush=True)
        for offset, count in asset["windows"]:
            target, command = readback(directory, f"asset_{index}_{offset:08x}", asset["address"]+offset, count)
            commands.append(command)
            samples.append((target, raw[offset:offset+count]))
    after_frames = None
    if "live_frames" in symbols:
        after_frames, command = readback(directory, "after_frames", symbols["live_frames"], 4)
        commands.append(command)
    jlink(args, directory, "verify_asset_boundaries", commands)
    for path, expected in samples:
        actual = path.read_bytes()
        if digest(actual) != digest(expected):
            raise RuntimeError(f"Readback checksum failed: {path}; ready flag remains zero")
    if after_frames:
        deadline = time.monotonic()+3
        attempt = 0
        while ((word(after_frames)-before["frames"]) & 0xffffffff) == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("Camera frame counter did not advance; ready flag remains zero")
            time.sleep(0.1)
            after_frames, command = readback(directory, "after_frames", symbols["live_frames"], 4)
            jlink(args, directory, f"verify_camera_{attempt}", [command], 3)
            attempt += 1
    if not args.no_enable:
        final_symbols = load_symbols(args.elf, args.nm)
        if any(final_symbols[name] != symbols[name] for name in ("live_assets_ready", "live_pipeline_state", "Reset_Handler")):
            raise RuntimeError("Live ELF changed during upload; ready flag remains zero")
        ready_address = final_symbols["live_assets_ready"]
        ready_path, ready_command = readback(directory, "ready_enabled", ready_address, 4)
        jlink(args, directory, "enable", [f"w4 0x{ready_address:08x} 1", ready_command])
        if word(ready_path) != 1:
            raise RuntimeError("Asset samples verified, but live_assets_ready write did not read back as 1")
    report = {"uploaded_bytes": uploaded, "skipped_bytes": skipped, "skipped_chunks": skipped_chunks,
              "resume_verification": "Fresh first/last128-byte readbacks per skipped chunk; not a full-chunk hash" if args.resume else None,
              "verified_windows": len(samples),
              "sampled_readback_bytes": sum(len(expected) for _, expected in samples),
              "elapsed_seconds": round(time.monotonic()-started, 2), "ready_enabled": not args.no_enable,
              "verify_only": args.verify_only,
              "camera_frames_advanced": ((word(after_frames)-before["frames"]) & 0xffffffff) if after_frames else None}
    (directory / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Asset upload failed: {error}", file=sys.stderr)
        raise SystemExit(1)
