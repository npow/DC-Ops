"""Read one completed RGB565 camera buffer for host-side diagnostics."""
import sys
from pathlib import Path
from device import ARTIFACTS, jlink, symbols
from PIL import Image

addresses = symbols()
name = next(name for name in addresses if 'rgb565' in name)
raw_path = ARTIFACTS / 'live_camera_rgb565.bin'
log = jlink(['h', f'savebin {raw_path} 0x{addresses[name]:08x} 0x25800', 'g'], 'live_camera_capture.log')
raw = raw_path.read_bytes()
if len(raw) != 320*240*2:
    raise RuntimeError('Unexpected camera frame size')
rgb = bytearray()
for offset in range(0, len(raw), 2):
    pixel = raw[offset] | raw[offset+1] << 8
    rgb.extend(((pixel >> 11) * 255 // 31, ((pixel >> 5) & 63) * 255 // 63, (pixel & 31) * 255 // 31))
path = ARTIFACTS / 'live_camera.png'
Image.frombytes('RGB', (320,240), bytes(rgb)).save(path)
print(path)
