"""Export assets, build the portable host test, and record frozen parity evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "artifacts/translator/frontend"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-assets", action="store_true")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    if not args.reuse_assets:
        subprocess.run([sys.executable, str(ROOT / "export_assets.py")], check=True)
    executable = OUT / ("host_test_sanitize" if args.sanitize else "host_test")
    command = [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"]
    if args.sanitize:
        command += ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    subprocess.run(command + [str(ROOT / "frontend.cpp"), str(ROOT / "host_test.cpp"), "-o", str(executable)], check=True)
    result = subprocess.run([str(executable), str(OUT)], check=True, text=True, capture_output=True)
    report = json.loads(result.stdout)
    report["sanitizers"] = args.sanitize
    report["manifest_sha256"] = hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest()
    report["sources"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["frontend.h", "frontend.cpp", "host_test.cpp", "export_assets.py"]}
    (OUT / ("validation_sanitize.json" if args.sanitize else "validation.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
