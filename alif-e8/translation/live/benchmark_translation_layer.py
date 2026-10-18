#!/usr/bin/env python3
"""Measure the frozen encoder layer on CPU and U85 through the live benchmark service."""
import argparse
import array
import hashlib
import json
import math
import re
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

from direct_upload import DirectConnection
from load_assets import ARTIFACTS, NM, load_symbols

U32 = ["status", "error", "target", "ms", "prepare_ms", "output_address", "output_bytes",
       "cpu_runs", "npu_runs", "cpu_hash", "npu_hash", "compare_valid", "nonfinite"]
U64 = ["cycles", "pmu_cycles", "pmu_active", "pmu_mac", "pmu_idle"]
FLOAT = ["mean_abs", "max_abs", "rmse", "stage_speedup"]


def read32(link, address, count=1):
    values = list(link.memory_read32(address, count))
    if len(values) != count:
        raise RuntimeError(f"Short read at0x{address:08x}")
    return values


def write32(link, address, value):
    if link.memory_write32(address, [value & 0xffffffff]) != 4:
        raise RuntimeError(f"Short write at0x{address:08x}")


def read_bytes(link, address, count):
    if address%4 == 0 and count%4 == 0:
        return struct.pack("<"+"I"*(count//4), *read32(link, address, count//4))
    raw = bytes(link.memory_read8(address, count))
    if len(raw) != count:
        raise RuntimeError("Short byte read")
    return raw


def driver_layout(elf, nm):
    sizes = subprocess.check_output([str(nm), "-S", "-n", str(elf)], text=True)
    records = [line.split() for line in sizes.splitlines()]
    pools = [r for r in records if len(r) == 4 and r[3].endswith("9fast_poolE") and int(r[1], 16) == 384*1024]
    drivers = [r for r in records if len(r) == 4 and r[3].endswith("4npusE")]
    if len(pools) != 1 or len(drivers) != 1:
        raise RuntimeError("Cannot uniquely resolve fast scratch and two-driver array from ELF")
    readelf = nm.with_name("arm-none-eabi-readelf")
    process = subprocess.Popen([str(readelf), "--debug-dump=info", str(elf)], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True)
    buffer, depth, found = [], None, None
    try:
        for line in process.stdout:
            tag = re.match(r" <(\d+)><[0-9a-f]+>:.*\((DW_TAG_\w+)\)", line)
            if buffer and tag and int(tag[1]) <= depth:
                block = "".join(buffer)
                if re.search(r"DW_AT_name[^\n]*: ethosu_driver\s*\n", block):
                    size = re.search(r"DW_AT_byte_size\s*:\s*(\d+)", block)
                    job = re.search(r"DW_AT_name[^\n]*:\s*job\s*\n(?:(?!DW_TAG_)[^\n]*\n)*?[^\n]*DW_AT_data_member_location\s*:\s*(\d+)", block)
                    if size and job:
                        found = int(size[1]), int(job[1])
                        break
                buffer = []
            if tag and tag[2] == "DW_TAG_structure_type" and not buffer:
                depth = int(tag[1]); buffer = [line]
            elif buffer:
                buffer.append(line)
    finally:
        process.terminate()
        process.wait()
    if not found or int(drivers[0][1], 16) != found[0]*2:
        raise RuntimeError("Driver layout is missing or array size disagrees with DWARF")
    return {"fast_pool_address": int(pools[0][0], 16), "fast_pool_bytes": int(pools[0][1], 16),
            "driver_address": int(drivers[0][0], 16), "driver_bytes": found[0], "job_offset": found[1]}


def idle(link, symbols, layout):
    if read32(link, symbols["live_benchmark_command"])[0]:
        return False
    if read32(link, symbols["live_benchmark_status"])[0] in (1, 2, 3, 4):
        return False
    for i in range(2):
        state = link.memory_read8(layout["driver_address"]+i*layout["driver_bytes"]+layout["job_offset"], 1)
        if len(state) != 1 or state[0] != 0:
            return False
    return True


def snapshot(link, symbols):
    state = {name: read32(link, symbols["live_benchmark_"+name])[0] for name in U32}
    for name in U64:
        low, high = read32(link, symbols["live_benchmark_"+name], 2)
        state[name] = low+(high << 32)
    for name in FLOAT:
        state[name] = struct.unpack("<f", struct.pack("<I", read32(link, symbols["live_benchmark_"+name])[0]))[0]
    return state


def floats(raw):
    values = array.array("f"); values.frombytes(raw)
    if sys.byteorder != "little": values.byteswap()
    return values


def compare(actual, reference):
    if len(actual) != 8192 or len(reference) != 8192:
        raise RuntimeError("Quality comparison requires8192 elements")
    if any(not math.isfinite(x) for x in actual) or any(not math.isfinite(x) for x in reference):
        raise RuntimeError("Nonfinite benchmark output")
    delta = [abs(a-b) for a, b in zip(actual, reference)]
    return {"elements": 8192, "mae": sum(delta)/8192, "max_absolute_error": max(delta),
            "rmse": math.sqrt(sum(x*x for x in delta)/8192)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, default=ARTIFACTS / "flashed_live.elf")
    parser.add_argument("--nm", type=Path, default=NM)
    parser.add_argument("--speed", type=int, default=15000)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--take-paused-mode", action="store_true", help="Temporarily own paused manual mode2 and restore it afterward")
    parser.add_argument("--cpu-hz", type=int, default=400000000)
    parser.add_argument("--npu-hz", type=int, default=400000000)
    parser.add_argument("--max-mae", type=float, default=0.05)
    parser.add_argument("--max-error", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    symbols = load_symbols(args.elf, args.nm)
    required = ["live_benchmark_enabled", "live_benchmark_command", "ethosu_fast_scratch", "ethosu_fast_scratch_size"]
    required += ["live_benchmark_"+n for n in U32+U64+FLOAT]
    if any(name not in symbols for name in required):
        raise RuntimeError("Actual live ELF lacks required benchmark symbols")
    layout = driver_layout(args.elf, args.nm)
    index = json.loads((ARTIFACTS / "spatial_staging/pipeline_index.json").read_text())["stages"][0]
    paths = [("cpu", ARTIFACTS / "staging/encoder_0_cpu_fp32.pte", 0xA3400000),
             ("npu", ARTIFACTS / "spatial_staging/encoder_0_u85.pte", 0xA1300000+index["offset"])]
    models = [{"target": target, "file": str(path), "address": address, "bytes": path.stat().st_size,
               "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for target, path, address in paths]
    if models[1]["bytes"] != index["bytes"] or models[1]["source_sha256"] != index["sha256"]:
        raise RuntimeError("Spatial encoder0 does not match its packed stage index")
    inputs = [(ARTIFACTS / f"staging/encoder_0_input_{i}.bin").read_bytes() for i in range(2)]
    if list(map(len, inputs)) != [32768, 128]: raise RuntimeError("Frozen input sizes differ")
    frozen_hash = 2166136261
    for byte in b"".join(inputs): frozen_hash = ((frozen_hash ^ byte)*16777619) & 0xffffffff
    reference = floats((ARTIFACTS / "staging/encoder_0_fp32_reference.bin").read_bytes())
    report = {"models": models, "layout": layout, "input_sha256": [hashlib.sha256(x).hexdigest() for x in inputs],
              "fixture_fnv1a32": frozen_hash, "warmup_per_target": 1, "timed_per_target": 3,
              "label": "FP32 CPU vs A16W8 spatial-FFN U85, same frozen encoder layer semantics",
              "model_device_verification": "First/last128-byte windows; source SHA values are host file hashes",
              "runs": [], "complete": False}
    if args.dry_run:
        print(json.dumps(report, indent=2)); return
    directory = ARTIFACTS / "translation_layer_benchmark" / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    connection = DirectConnection(); owned_mode = False; old_fast = None
    try:
        connection.connect(args); link = connection.link
        if read32(link, symbols["live_assets_ready"])[0] != 1 or read32(link, symbols["live_pipeline_state"])[0] != 4:
            raise RuntimeError("Load baseline assets before enabling the initialized live pipeline")
        if read32(link, 0xe000ed08)[0] != 0x80200000 or read32(link, 0x80200004)[0] != (symbols["Reset_Handler"] | 1):
            raise RuntimeError("Target vectors differ from actual supplied ELF")
        mode = read32(link, symbols["live_benchmark_enabled"])[0]
        if mode == 2 and args.take_paused_mode:
            if any(read32(link, symbols[name])[0] for name in ("manual_translation_command",) if name in symbols):
                raise RuntimeError("Manual command is pending")
            if "manual_translation_status" in symbols and read32(link, symbols["manual_translation_status"])[0] == 1:
                raise RuntimeError("Manual translation is running")
            write32(link, symbols["live_benchmark_enabled"], 1); owned_mode = True
        elif mode != 1:
            raise RuntimeError("Reserve benchmark mode1 first, or explicitly use --take-paused-mode for paused mode2")
        if not idle(link, symbols, layout): raise RuntimeError("Benchmark or NPU driver is not idle")
        for model in models:
            raw = Path(model["file"]).read_bytes()
            for offset in (0, len(raw)-128):
                if read_bytes(link, model["address"]+offset, 128) != raw[offset:offset+128]:
                    raise RuntimeError(f"Mapped {model['target']} model sample differs from host source")
        for i, address in enumerate((0xA3900000, 0xA3910000)):
            if read_bytes(link, address, len(inputs[i])) != inputs[i]: raise RuntimeError("Full frozen input readback differs")
        old_fast = (read32(link, symbols["ethosu_fast_scratch"])[0], read32(link, symbols["ethosu_fast_scratch_size"])[0])
        write32(link, symbols["ethosu_fast_scratch"], layout["fast_pool_address"])
        write32(link, symbols["ethosu_fast_scratch_size"], layout["fast_pool_bytes"])
        outputs = {}
        for pair in range(4):
            for target in (1, 2):
                if not idle(link, symbols, layout): raise RuntimeError("Driver ceased to be idle between trials")
                before = snapshot(link, symbols)
                frame_before = read32(link, symbols["live_frames"])[0]
                write32(link, symbols["live_benchmark_status"], 0)
                write32(link, symbols["live_benchmark_command"], target)
                started = time.monotonic()
                while True:
                    status = read32(link, symbols["live_benchmark_status"])[0]
                    if status == 0xe0: raise RuntimeError(f"Benchmark firmware error={read32(link, symbols['live_benchmark_error'])[0]}")
                    if status == 5 and read32(link, symbols["live_benchmark_command"])[0] == 0: break
                    if time.monotonic()-started > args.timeout: raise RuntimeError(f"Target{target} exceeded{args.timeout}s")
                    time.sleep(0.05)
                state = snapshot(link, symbols)
                expected_output = 0x02210000 if target == 1 else 0x02218000
                key = "cpu" if target == 1 else "npu"
                if state["error"] or state["target"] != target or state["output_bytes"] != 32768 or state["output_address"] != expected_output:
                    raise RuntimeError(f"Invalid completion metadata: {state}")
                if state[key+"_runs"] != before[key+"_runs"]+1 or state[key+"_hash"] != frozen_hash:
                    raise RuntimeError("Run count or frozen firmware input hash differs")
                raw = read_bytes(link, state["output_address"], 32768)
                (directory / f"{key}_{pair}.bin").write_bytes(raw)
                values = floats(raw); quality = compare(values, reference)
                if quality["mae"] > args.max_mae or quality["max_absolute_error"] > args.max_error:
                    raise RuntimeError(f"{key} output fails bounded numeric comparison: {quality}")
                outputs[target, pair] = values
                frame_after = read32(link, symbols["live_frames"])[0]
                state.update(target_name=key, warmup=pair == 0, pair=pair,
                    host_wait_seconds=time.monotonic()-started, reference_quality=quality,
                    execute_ms_from_cycles=state["cycles"]*1000/args.cpu_hz,
                    camera_frames_advanced=(frame_after-frame_before) & 0xffffffff)
                if target == 2:
                    if not state["pmu_cycles"] or not state["pmu_active"]: raise RuntimeError("NPU run lacks measured PMU activity")
                    if not state["compare_valid"] or state["nonfinite"]: raise RuntimeError("Firmware paired-output comparison is invalid")
                    state["active_duty_percent"] = 100*(state["pmu_active"]/args.npu_hz)/(state["cycles"]/args.cpu_hz)
                    state["mac_active_fraction_percent"] = 100*state["pmu_mac"]/state["pmu_cycles"]
                    state["paired_quality"] = compare(values, outputs[1, pair])
                    if state["paired_quality"]["mae"] > args.max_mae or state["paired_quality"]["max_absolute_error"] > args.max_error:
                        raise RuntimeError("CPU/NPU paired output difference exceeds numeric limits")
                report["runs"].append(state)
                (directory / "result.json").write_text(json.dumps(report, indent=2)+"\n")
                print(f"{key} {'warmup' if pair == 0 else 'timed'+str(pair)} execute={state['ms']}ms prepare={state['prepare_ms']}ms MAE={quality['mae']:.6g}", flush=True)
        medians = {key: statistics.median(r["execute_ms_from_cycles"] for r in report["runs"] if not r["warmup"] and r["target_name"] == key) for key in ("cpu", "npu")}
        if not any(run["camera_frames_advanced"] for run in report["runs"]):
            raise RuntimeError("Camera frame counter never advanced during the benchmark")
        report.update(median_execute_ms=medians, speedup=medians["cpu"]/medians["npu"], complete=True,
                      cpu_hz=args.cpu_hz, npu_hz=args.npu_hz, quality_limits={"mae": args.max_mae, "max_error": args.max_error})
        print(json.dumps({"median_execute_ms": medians, "speedup": report["speedup"], "report": str(directory / "result.json")}, indent=2))
    except Exception as error:
        report["error"] = str(error); raise
    finally:
        try:
            if connection.link is not None:
                if idle(connection.link, symbols, layout):
                    if old_fast is not None:
                        write32(connection.link, symbols["ethosu_fast_scratch_size"], 0)
                        write32(connection.link, symbols["ethosu_fast_scratch"], old_fast[0])
                        write32(connection.link, symbols["ethosu_fast_scratch_size"], old_fast[1])
                    if owned_mode: write32(connection.link, symbols["live_benchmark_enabled"], 2)
                    report["idle_state_restored"] = True
                else:
                    report["idle_state_restored"] = False
        except Exception as error:
            report["cleanup_error"] = str(error)
            report["idle_state_restored"] = False
        finally:
            connection.close()
            (directory / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        if not report.get("idle_state_restored"):
            report["complete"] = False
            (directory / "result.json").write_text(json.dumps(report, indent=2)+"\n")
            raise RuntimeError("Benchmark is still active or idle cleanup failed; benchmark mode left reserved")


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"Translation benchmark failed: {error}", file=sys.stderr); raise SystemExit(1)
