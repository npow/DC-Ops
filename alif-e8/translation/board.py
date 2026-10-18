"""Bounded J-Link transfers for the translator hardware harness."""
import json
import os
import re
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts/translator"
ELF = ARTIFACTS / "flashed_benchmark.elf"
NM = "/Users/npow/.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin/arm-none-eabi-nm"


def symbols():
    output = subprocess.check_output([NM, "-n", str(ELF)], text=True)
    return {parts[2]: int(parts[0], 16) for line in output.splitlines()
            if len(parts := line.split()) == 3}


def jlink(commands, log_name="hardware_jlink.log", timeout=120):
    cleanup = [f"w4 0x{0xe0002008+i*4:08x} 0" for i in range(8)]
    script = "\n".join(["h", "exec SetEnableFlashbank 0xA0000000=0",
        "exec SetAllowFlashCache=0", *commands, *cleanup, "g", "exit", ""])
    result = subprocess.run(["JLinkExe", "-device", "AE822FA0E5597LS0_M55_HP", "-if", "SWD",
        "-speed", os.environ.get("DCOPS_SWD_KHZ", "4000"), "-autoconnect", "1"], input=script, capture_output=True,
        text=True, timeout=timeout)
    log = result.stdout+result.stderr
    (ARTIFACTS / log_name).write_text(log)
    if result.returncode or any(message in log for message in (
        "Cannot connect", "Error while", "Could not read memory", "Failed to download")):
        raise RuntimeError(log[-4000:])
    return log


def status():
    addresses = symbols()
    fields = ["tr_status", "tr_command", "tr_error", "tr_model_index", "tr_execute_cycles",
              "tr_pmu_active", "tr_pmu_mac", "tr_pmu_idle", "tr_npu_product", "tr_npu_macs",
              "tr_output_address", "tr_output_bytes", "tr_milliseconds", "tr_execute_ms"]
    fields = [name for name in fields if name in addresses]
    commands = ["regs", "mem32 0xE000ED08 1"] + [
        f"savebin {ARTIFACTS / (name+'.bin')} 0x{addresses[name]:08x} 0x4" for name in fields]
    commands += [f"savebin {ARTIFACTS / 'tr_pmu_cycles.bin'} 0x{addresses['tr_pmu_cycles']:08x} 0x8",
                 f"savebin {ARTIFACTS / 'tr_log.bin'} 0x{addresses['dc_log']:08x} 0x4000"]
    debug_log = jlink(commands, "status_jlink.log")
    vector = re.search(r"E000ED08 = ([0-9A-Fa-f]+)", debug_log)
    if vector is None or int(vector.group(1), 16) != 0x80200000:
        raise RuntimeError("Translator is not running: vector table does not match the benchmark image")
    values = {name: struct.unpack("<I", (ARTIFACTS / (name+'.bin')).read_bytes())[0] for name in fields}
    values["tr_pmu_cycles"] = struct.unpack("<Q", (ARTIFACTS / 'tr_pmu_cycles.bin').read_bytes())[0]
    log = (ARTIFACTS / 'tr_log.bin').read_bytes().decode(errors='replace').replace('\0','')
    print(log[-4000:])
    print(json.dumps(values, indent=2))
    (ARTIFACTS / 'hardware_status.json').write_text(json.dumps(values, indent=2)+'\n')
    return values


if __name__ == "__main__":
    status()
