"""Run identical frozen input tensors on the E8 and collect timing/PMU counters."""
import argparse
import contextlib
import io
import json
import statistics
import shutil
import time

import numpy as np

from board import ARTIFACTS, jlink, status, symbols


def get_status():
    with contextlib.redirect_stdout(io.StringIO()):
        return status()


def wait_status(expected, timeout=60):
    start = time.monotonic()
    while time.monotonic()-start < timeout:
        value = get_status()
        if value["tr_status"] == expected:
            return value
        if value["tr_status"] >= 0xe0:
            raise RuntimeError(value)
        time.sleep(1)
    raise TimeoutError(value)


def load_model(name, target, reuse_model=False):
    addresses = symbols()
    suffix = "u85" if target == "npu" else "cpu_fp32"
    path = ARTIFACTS / (name+"_"+suffix+".pte")
    transfer = ARTIFACTS / (name+"_"+suffix+".bin")
    shutil.copyfile(path, transfer)
    commands = ([] if reuse_model else [f"loadfile {transfer} 0xa0000000 noreset"]) + [
        f"w4 0x{addresses['tr_model_address']:08x} 0xa0000000",
        f"w4 0x{addresses['tr_model_size']:08x} 0x{path.stat().st_size:x}",
        f"w4 0x{addresses['tr_model_index']:08x} 0",
        f"w4 0x{addresses['tr_command']:08x} 1"]
    jlink(commands, "load_"+name+"_"+target+".log", timeout=240)
    return wait_status(5)


def set_inputs(name, data):
    addresses = symbols()
    commands = []
    count = 2 if name == "encoder" else 4
    for i in range(count):
        tensor = data[f"{name}_input_{i}"]
        path = ARTIFACTS / (name+f"_input_{i}.bin")
        tensor.astype("<f4").tofile(path)
        pointer = 0xa3700000+i*0x10000
        commands += [f"loadfile {path} 0x{pointer:08x} noreset",
            f"w4 0x{addresses['tr_input_address']+i*4:08x} 0x{pointer:08x}",
            f"w4 0x{addresses['tr_input_bytes']+i*4:08x} 0x{tensor.nbytes:x}"]
    jlink(commands, "inputs_"+name+".log")


def execute(name, target, index):
    addresses = symbols()
    jlink([f"w4 0x{addresses['tr_command']:08x} 2"], "execute_"+name+".log")
    time.sleep(1 if target == "npu" else 30)
    result = wait_status(6, timeout=90)
    path = ARTIFACTS / f"{name}_{target}_output_{index}.bin"
    jlink([f"savebin {path} 0x{result['tr_output_address']:08x} 0x{result['tr_output_bytes']:x}"],
          "output_"+name+".log")
    output = np.fromfile(path, dtype="<f4")
    return result, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["npu", "cpu"], required=True)
    parser.add_argument("--model", choices=["encoder", "decoder", "both"], default="both")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--reuse-model", action="store_true", help="Reuse model bytes already in PSRAM")
    args = parser.parse_args()
    data = np.load(ARTIFACTS / "hardware_fixture.npz")
    reports = {}
    for name in ([args.model] if args.model != "both" else ["encoder", "decoder"]):
        print("LOADING", args.target, name, flush=True)
        if args.reuse_model and args.model == "both":
            raise ValueError("--reuse-model requires one model")
        load_model(name, args.target, args.reuse_model)
        set_inputs(name, data)
        values = []
        suffix = "a16w8" if args.target == "npu" else "fp32"
        reference = data[name+"_"+suffix+"_output"].flatten()
        for i in range(args.samples+1):
            stats, output = execute(name, args.target, i)
            mae = float(np.abs(output-reference).mean())
            maximum = float(np.abs(output-reference).max())
            if not np.isfinite(output).all():
                raise AssertionError("Non-finite hardware output")
            if name == "decoder" and int(output.argmax()) != int(reference.argmax()):
                raise AssertionError(f"Token mismatch: {output.argmax()} vs {reference.argmax()}")
            row = {**stats, "mae": mae, "max_abs_error": maximum, "warmup": i == 0,
                   "output_elements": output.size}
            print(json.dumps(row), flush=True)
            values.append(row)
        measured = values[1:]
        reports[name] = {"denominator": len(measured), "warmups": 1, "samples": values,
            "median_ms": statistics.median(r["tr_execute_ms"] for r in measured),
            "precision": "A16W8" if args.target == "npu" else "FP32",
            "cpu_kernel_placement": "cached MRAM, compiler optimize=speed",
            "timing_scope": "ExecuTorch method execution; excludes host debugger transfers, tokenization, camera and display"}
    path = ARTIFACTS / ("benchmark_"+args.target+".json")
    path.write_text(json.dumps(reports, indent=2)+"\n")
    print(path, flush=True)


if __name__ == "__main__":
    main()
