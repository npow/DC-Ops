#!/usr/bin/env python3
"""Stage the existing CPU layer baseline and frozen inputs without enabling inference."""
import sys

import direct_upload
import load_assets


def main():
    load_assets.ASSETS = [
        ("staging/encoder_0_cpu_fp32.pte", 0xA3400000, 4219536),
        ("staging/encoder_0_input_0.bin", 0xA3900000, 32768),
        ("staging/encoder_0_input_1.bin", 0xA3910000, 128),
    ]
    if not any(arg == "--elf" or arg.startswith("--elf=") for arg in sys.argv[1:]):
        sys.argv.extend(["--elf", str(load_assets.ARTIFACTS / "flashed_live.elf")])
    if "--via-staging" not in sys.argv:
        sys.argv.append("--via-staging")
    if "--no-enable" not in sys.argv:
        sys.argv.append("--no-enable")
    direct_upload.main()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Benchmark asset upload failed: {error}", file=sys.stderr)
        raise SystemExit(1)
