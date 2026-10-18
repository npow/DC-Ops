"""Compare the same frozen NPU workload from SRAM and external PSRAM."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from board import ARTIFACTS, jlink, status, symbols
from benchmark import get_status, wait_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", choices=["sram", "psram"], required=True)
    parser.add_argument("--program", type=Path, default=ARTIFACTS / "memory_probe_u85.pte")
    parser.add_argument("--fixture", type=Path, default=ARTIFACTS / "memory_probe.npz")
    args = parser.parse_args()
    addresses = symbols()
    program = args.program
    transfer = ARTIFACTS / "memory_probe_model.bin"
    transfer.write_bytes(program.read_bytes())
    data = np.load(args.fixture)
    inputs = [data["input"]] if "input" in data else [data[f"input_{i}"] for i in range(4) if f"input_{i}" in data]
    pointer = 0x02100000 if args.memory == "sram" else 0xa3000000
    jlink([f"loadfile {transfer} 0x{pointer:x} noreset",
           f"w4 0x{addresses['tr_model_address']:x} 0x{pointer:x}",
           f"w4 0x{addresses['tr_model_size']:x} 0x{program.stat().st_size:x}",
           f"w4 0x{addresses['tr_model_index']:x} 0",
           f"w4 0x{addresses['tr_command']:x} 1"], "probe_load.log")
    wait_status(5)
    commands = []
    for i, value in enumerate(inputs):
        input_file = ARTIFACTS / f"memory_probe_input_{i}.bin"
        value.astype("<f4").tofile(input_file)
        input_pointer = 0xa3700000+i*0x10000
        commands += [f"loadfile {input_file} 0x{input_pointer:x} noreset",
                     f"w4 0x{addresses['tr_input_address']+i*4:x} 0x{input_pointer:x}",
                     f"w4 0x{addresses['tr_input_bytes']+i*4:x} 0x{value.nbytes:x}"]
    commands += [f"w4 0x{addresses['tr_command']:x} 2"]
    jlink(commands, "probe_execute.log")
    time.sleep(1)
    result = wait_status(6, timeout=25)
    output_file = ARTIFACTS / (program.stem+"_"+args.memory+"_output.bin")
    jlink([f"savebin {output_file} 0x{result['tr_output_address']:x} 0x{result['tr_output_bytes']:x}"],
          "probe_output.log")
    output = np.fromfile(output_file, dtype="<f4")
    reference = data["reference" if "reference" in data else "output"].flatten()
    result.update(memory=args.memory, mae=float(np.abs(output-reference).mean()),
                  max_error=float(np.abs(output-reference).max()), output_elements=output.size)
    result["program"] = str(program)
    (ARTIFACTS / (program.stem+"_"+args.memory+".json")).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
