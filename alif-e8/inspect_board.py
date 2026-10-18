"""Read E8 status and optionally the LCD/camera buffers without changing firmware."""
import argparse
import json
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLCHAIN = Path('/Users/npow/.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin')
ELF = ROOT / 'out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.elf'


def symbols():
    output = subprocess.check_output([str(TOOLCHAIN / 'arm-none-eabi-nm'), '-n', str(ELF)], text=True)
    return {parts[2]: int(parts[0], 16) for line in output.splitlines()
            if len(parts := line.split()) == 3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', action='store_true')
    args = parser.parse_args()
    addresses = symbols()
    reads = [('dc_log_position', 4), ('dc_log', 16384), ('dc_status', 4),
             ('dc_frame_count', 4), ('dc_detection_count', 4), ('dc_inference_ms', 4),
             ('dc_lcd_front', 4), ('dc_lcd_vsyncs', 4), ('dc_lcd_presents', 4),
             ('dc_lcd_timeouts', 4), ('dc_camera_checksum', 4), ('dc_detections', 12 * 24)]
    if args.images:
        reads += [('lcd_frame', 2 * 480 * 800 * 2), ('dc_camera_frame', 320 * 240 * 2)]
    script = 'h\nregs\n' + ''.join(
        f'savebin {ROOT / "artifacts" / (name + ".bin")} 0x{addresses[name]:08x} 0x{size:x}\n'
        for name, size in reads) + ''.join(f'w4 0x{0xe0002008+i*4:08x} 0\n' for i in range(8)) + 'g\nexit\n'
    result = subprocess.run(['JLinkExe', '-device', 'AE822FA0E5597LS0_M55_HP',
        '-if', 'SWD', '-speed', '4000', '-autoconnect', '1'], input=script,
        capture_output=True, text=True, timeout=120)
    (ROOT / 'artifacts/inspect_jlink.log').write_text(result.stdout + result.stderr)
    values = {name: struct.unpack('<I', (ROOT / 'artifacts' / (name + '.bin')).read_bytes())[0]
              for name, size in reads if size == 4}
    position = values['dc_log_position']
    ring = (ROOT / 'artifacts/dc_log.bin').read_bytes()
    log = ring[:position] if position < len(ring) else ring[position % len(ring):] + ring[:position % len(ring)]
    print(log.decode(errors='replace').replace('\x00', '')[-12000:])
    (ROOT / 'artifacts/board.log').write_bytes(log)
    print(json.dumps(values, indent=2))
    (ROOT / 'artifacts/board_status.json').write_text(json.dumps(values, indent=2))
    if args.images:
        import numpy as np
        from PIL import Image
        lcd = np.fromfile(ROOT / 'artifacts/lcd_frame.bin', dtype='<u2').reshape(2, 800, 480)[values['dc_lcd_front']]
        rgb = np.stack([((lcd >> 11) & 31) * 255 // 31,
                        ((lcd >> 5) & 63) * 255 // 63, (lcd & 31) * 255 // 31], -1).astype('uint8')
        Image.fromarray(rgb).save(ROOT / 'artifacts/lcd.png')
        pixels = np.fromfile(ROOT / 'artifacts/dc_camera_frame.bin', dtype='<u2').reshape(240, 320)
        rgb = np.stack([((pixels >> 11) & 31) * 255 // 31,
                        ((pixels >> 5) & 63) * 255 // 63, (pixels & 31) * 255 // 31], -1).astype('uint8')
        Image.fromarray(rgb).save(ROOT / 'artifacts/camera.png')


if __name__ == '__main__':
    main()
