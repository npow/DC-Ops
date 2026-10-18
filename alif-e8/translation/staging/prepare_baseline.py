"""Describe and validate the already-exported matching CPU/NPU layer baseline."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from executorch.runtime import Runtime

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts/translator"
STAGING = ARTIFACTS / "staging"
SPATIAL = ARTIFACTS / "spatial_staging"


def main():
    fixture = np.load(STAGING / "hardware_fixture.npz")
    inputs = [fixture[f"encoder_input_{i}"] for i in range(2)]
    reference = fixture["encoder_fp32_output"]
    reference.astype("<f4").tofile(STAGING / "encoder_0_fp32_reference.bin")
    np.savez(STAGING / "encoder_0_cpu_fixture.npz", input_0=inputs[0], input_1=inputs[1], output=reference)
    rows = {}
    for name, path in [("cpu", STAGING / "encoder_0_cpu_fp32.pte"), ("npu", SPATIAL / "encoder_0_u85.pte")]:
        raw = path.read_bytes()
        rows[name] = {"file": str(path), "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()}
    cpu_metadata = json.loads((STAGING / "encoder_0_cpu_fp32.json").read_text())
    npu_metadata = json.loads((SPATIAL / "memory_manifest.json").read_text())["encoder_0"]
    rows["cpu"].update(precision="FP32", planned_bytes=max(cpu_metadata["planned_buffers"]),
                       suggested_program_address="0xA3400000", suggested_planned_address="0x025A0000")
    rows["npu"].update(precision="A16W8", planned_bytes=max(npu_metadata["planned_buffers"]),
                       delegates=npu_metadata["delegates"], staging_bytes=2560*1024,
                       scratch_bytes=npu_metadata["scratch_bytes"], fast_scratch_bytes=npu_metadata["fast_scratch_bytes"],
                       ffn_rewrite="spatial1x1convolutions")
    program = Runtime.get().load_program(STAGING / "encoder_0_cpu_fp32.pte")
    method = program.load_method("forward")
    actual = method.execute([torch.from_numpy(x) for x in inputs])[0].numpy()
    absolute = np.abs(actual-reference)
    report = {
        "architecture": "OPUS-MT Tiny English-Spanish encoder layer 0, sequence32 hidden256",
        "input_text": "Please close the door", "models": rows,
        "input_shapes": [list(x.shape) for x in inputs],
        "input_bytes": [x.nbytes for x in inputs], "input_dtype": "float32 contiguous",
        "output_shape": list(reference.shape), "output_bytes": reference.nbytes,
        "cpu_fixture": str(STAGING / "encoder_0_cpu_fixture.npz"),
        "npu_fixture": str(STAGING / "spatial_translator/encoder0_fixture.npz"),
        "host_cpu_reference_check": {"mean_absolute_error": float(absolute.mean()),
                                     "max_absolute_error": float(absolute.max()),
                                     "output_elements": int(reference.size)},
        "hardware_measurement_protocol": {
            "warmup_runs": 1, "timed_runs": 3, "report": "median and individual wall times",
            "cpu_load": "Read CPU PTE directly from PSRAM; it exceeds SRAM staging capacity",
            "npu_load": "Copy NPU PTE to SRAM staging before loading method",
            "timed_region": "Runner.execute only; record PTE load and staging copy separately",
            "speedup": "median CPU execute wall time divided by median NPU execute wall time",
            "label": "FP32 CPU versus A16W8 U85; quantization and acceleration both differ",
            "quality_note": "Spatial FFN candidate; report paired8192-element output errors alongside measured timing",
            "availability": "Do not interrupt the working live pipeline to run this baseline",
        },
        "npu_metrics": {
            "execution_ms": "DWT cycles/400000 (interval <10.7s), otherwise monotonic milliseconds",
            "npu_active_duty_percent": "100*(NPU_ACTIVE_cycles/NPU_clock_hz)/execute_wall_seconds",
            "mac_active_fraction_percent": "100*MAC_ACTIVE_cycles/total_NPU_cycles; not peak MAC utilization",
            "full_pipeline": "Also report frames, OCR ms, translator ms, tokens, and stage-copy/load time",
        },
        "end_to_end_validation": {
            "source_samples": str(ARTIFACTS / "samples.json"),
            "evaluation_denominator": 8,
            "record": ["OCR English text", "generated Spanish UTF8", "token IDs and EOS status",
                       "NPU counters before/after", "wall time and live frame advancement"],
            "functional_evidence": "Camera remains live and an OCR-derived translation appears in the corresponding screen box",
            "quality_policy": "Record arbitrary output and errors without requiring exact tokens or blocking visible proof",
            "known_limits": "Host PT2E 8/8 parity is not E8 translation accuracy; actual hardware outputs remain to be recorded",
        },
    }
    existing = ARTIFACTS / "encoder_0_u85_sram.json"
    if existing.exists():
        report["prior_npu_diagnostic_run"] = json.loads(existing.read_text())
        report["prior_npu_diagnostic_note"] = "Single non-quality-qualified run; no paired CPU timing or speedup yet"
    (STAGING / "baseline_plan.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
