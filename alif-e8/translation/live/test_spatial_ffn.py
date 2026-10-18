"""Test a padded two-input FFN with existing firmware, then restore live assets."""
import hashlib
import json
import struct
import time
from types import SimpleNamespace

import numpy as np

from device import ARTIFACTS, symbols
from direct_upload import DirectConnection
from repair_assets import expected_crc, scan


def main():
    firmware_sha = hashlib.sha256((ARTIFACTS / "flashed_live.elf").read_bytes()).hexdigest()
    if firmware_sha != "f13a768957285e2e4d05010e12fe341892fbcddf6952b16a9195fac0ea8b6de7":
        raise RuntimeError("This probe requires the original fixed-length benchmark firmware")
    folder = ARTIFACTS / "staging/offline_spatial_ffn_two_input"
    candidate = (folder / "ffn_spatial_two_input_u85.pte").read_bytes()
    fixture = np.load(folder / "fixture.npz")
    original = (ARTIFACTS / "staging/pipeline.bin").read_bytes()[:1038896]
    original_fixture = np.load(ARTIFACTS / "staging/encoder_0_cpu_fixture.npz")
    if len(candidate) > len(original):
        raise RuntimeError("Candidate exceeds the firmware's fixed staging length")
    staged = candidate.ljust(len(original), b"\0")
    payloads = [(0xa1300000, staged), (0xa3900000, fixture["input_0"].tobytes()),
                (0xa3910000, fixture["input_1"].tobytes())]
    if [len(raw) for _, raw in payloads] != [1038896, 32768, 128]:
        raise RuntimeError("Probe signature differs from the existing benchmark")
    directory = ARTIFACTS / "spatial_ffn_hardware" / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    table = symbols()
    connection = DirectConnection()
    connection.connect(SimpleNamespace(speed=15000))
    link = connection.link

    def read(name):
        return link.memory_read32(table[name], 1)[0]

    def write(name, value):
        if link.memory_write32(table[name], [value]) != 4:
            raise RuntimeError(f"Failed write to {name}")

    def upload(address, raw):
        if (read("live_asset_command") or read("live_integrity_command") or read("live_benchmark_command") or
                read("live_integrity_status") == 1 or read("live_benchmark_status") in (1, 2, 3, 4)):
            raise RuntimeError("A firmware service still owns staging; it was not overwritten")
        connection.write_bytes(table["live_model_staging"], raw, time.monotonic()+30)
        sequence = (read("live_asset_completed_sequence")+1) & 0xffffffff or 1
        for field, value in (("destination", address), ("bytes", len(raw)),
                             ("sequence", sequence), ("command", 1)):
            write("live_asset_"+field, value)
        deadline = time.monotonic()+15
        while read("live_asset_command") or read("live_asset_completed_sequence") != sequence:
            if time.monotonic() > deadline:
                raise RuntimeError("Asset copy did not finish; staging is still owned by firmware")
            time.sleep(.05)
        if read("live_asset_error"):
            raise RuntimeError("Asset copy failed")

    def verify(address, raw, label):
        crcs = expected_crc(raw)
        result = scan(connection, table, address, raw, struct.pack(f"<{len(crcs)}I", *crcs),
                      30, directory, label)
        if result["bad_count"]:
            raise RuntimeError(f"{label} has {result['bad_count']} corrupted blocks")
        return result

    def wait_for_main_loop():
        initial = read("live_frames")
        deadline = time.monotonic()+5
        while ((read("live_frames")-initial) & 0xffffffff) < 2:
            if time.monotonic() > deadline:
                raise RuntimeError("Camera frame progress did not prove return to the main loop")
            time.sleep(.02)

    touched = False
    completed = False
    owned = False
    benchmark_started = False
    report = {"firmware_sha256": hashlib.sha256((ARTIFACTS / "flashed_live.elf").read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(candidate).hexdigest(), "firmware_changed": False}
    try:
        if read("live_pipeline_state") != 4 or read("live_assets_ready") != 1:
            raise RuntimeError("Live pipeline is not initialized")
        if read("live_benchmark_enabled") or read("live_benchmark_command") or read("manual_translation_command"):
            raise RuntimeError("Another diagnostic already owns inference")
        if (read("live_asset_command") or read("live_integrity_command") or
                read("live_integrity_status") == 1 or read("live_benchmark_status") in (1, 2, 3, 4)):
            raise RuntimeError("A firmware diagnostic is still active")
        write("live_benchmark_enabled", 1)
        owned = True
        deadline = time.monotonic()+60
        busy = next(address for name, address in table.items() if name.endswith("4busyE"))
        while link.memory_read8(busy, 1)[0]:
            if time.monotonic() > deadline:
                raise RuntimeError("OCR did not finish its current pass")
            time.sleep(.1)
        wait_for_main_loop()
        # Both workers must have relinquished their shared buffers.
        request = link.memory_read32(0x02680008, 1)[0]
        response = link.memory_read32(0x02680080, 1)[0]
        if request and request != response:
            raise RuntimeError("HE still owns its input/output buffers")
        frame_before = read("live_frames")
        write("live_assets_ready", 0)
        write("live_pipeline_state", 2)
        touched = True
        report["verification"] = []
        for index, (address, raw) in enumerate(payloads):
            upload(address, raw)
            report["verification"].append(verify(address, raw, f"probe_{index}"))
        fast = next(address for name, address in table.items() if name.endswith("10fast_poolE") or name.endswith("9fast_poolE"))
        write("ethosu_fast_scratch", fast)
        write("ethosu_fast_scratch_size", 384*1024)
        write("live_benchmark_status", 0)
        write("live_benchmark_command", 2)
        benchmark_started = True
        deadline = time.monotonic()+30
        while read("live_benchmark_status") not in (5, 0xe0):
            if time.monotonic() > deadline:
                raise RuntimeError("Benchmark did not finish; inference remains paused")
            time.sleep(.05)
        fields = ["live_benchmark_status", "live_benchmark_error", "live_benchmark_ms",
                  "live_benchmark_prepare_ms", "live_benchmark_output_address", "live_benchmark_output_bytes"]
        report.update({name: read(name) for name in fields})
        if report["live_benchmark_status"] != 5 or report["live_benchmark_error"]:
            raise RuntimeError(f"Probe failed: {report}")
        wait_for_main_loop()
        completed = True
        if report["live_benchmark_output_bytes"] != 32768:
            raise RuntimeError("Probe output length differs")
        output = np.array(link.memory_read32(report["live_benchmark_output_address"], 8192), dtype="<u4").view("<f4").reshape(1, 32, 256)
        reference = fixture["output"]
        error = np.abs(output-reference)
        np.savez(directory / "output.npz", actual=output, reference=reference)
        report.update(elements=int(error.size), nonfinite=int((~np.isfinite(output)).sum()),
                      mean_abs_error=float(error.mean()), max_abs_error=float(error.max()),
                      row_mean_abs_error=error.mean(axis=(0, 2)).tolist(),
                      frames_during_probe=read("live_frames")-frame_before)
    finally:
        try:
            if touched:
                if benchmark_started and not completed:
                    raise RuntimeError("NPU still owns staging; live OCR was left paused")
                upload(0xa1300000, original)
                report["restore_verification"] = verify(0xa1300000, original, "restored_encoder0")
                for index, address in enumerate((0xa3900000, 0xa3910000)):
                    raw = original_fixture[f"input_{index}"].tobytes()
                    upload(address, raw)
                    verify(address, raw, f"restored_fixture_{index}")
                write("ethosu_fast_scratch", 0)
                write("ethosu_fast_scratch_size", 0)
                write("live_pipeline_state", 4)
                write("live_assets_ready", 1)
            if owned:
                write("live_benchmark_enabled", 0)
                report["live_ocr_restored"] = True
        finally:
            report["lcd_timeouts"] = read("translator_lcd_timeouts")
            report["lcd_errors"] = read("translator_lcd_errors")
            (directory / "result.json").write_text(json.dumps(report, indent=2)+"\n")
            connection.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
