"""Generate HE-only J-Link launch commands; never connect to or modify hardware."""
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ELF = ROOT / 'out/translator-he/DevKit-E8-HE/Debug/translator-he.elf'
OUTPUT = ROOT / 'artifacts/translator/he_boot.jlink'
NM = Path.home() / '.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin/arm-none-eabi-nm'


def main():
    data = ELF.read_bytes()
    if data[:6] != b'\x7fELF\x01\x01':
        raise ValueError('Expected ELF32 little-endian image')
    phoff = struct.unpack_from('<I', data, 28)[0]
    entsize, count = struct.unpack_from('<HH', data, 42)
    vector = None
    for index in range(count):
        kind, offset, _, physical, filesz, _, _, _ = struct.unpack_from('<8I', data, phoff + index * entsize)
        if kind == 1 and physical == 0x80100000 and filesz >= 8:
            vector = struct.unpack_from('<2I', data, offset)
    if vector is None:
        raise ValueError('No loadable HE vector table at 0x80100000')
    msp, reset = vector
    if not 0x20000000 <= msp <= 0x20040000 or not 0x80100000 <= reset < 0x80200000 or not reset & 1:
        raise ValueError(f'Invalid HE vectors: MSP={msp:#x}, Reset={reset:#x}')
    commands = ['h', f'loadfile "{ELF}"', 'w4 0xE000ED08 0x80100000', f'wreg MSP 0x{msp:08X}',
                'wreg CONTROL 0', 'wreg PRIMASK 0', 'wreg BASEPRI 0', 'wreg FAULTMASK 0',
                f'SetPC 0x{reset & ~1:08X}']
    commands += [f'w4 0x{address:08X} 0' for address in range(0xe0002008, 0xe0002028, 4)]
    commands += ['g', 'exit']
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text('\n'.join(commands) + '\n')
    symbols = {}
    result = subprocess.run([str(NM), '-n', str(ELF)], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2].startswith('he_worker_'):
            address = int(fields[0], 16)
            symbols[fields[2]] = {'local': hex(address), 'hp_global': hex(address - 0x20000000 + 0x58800000) if 0x20000000 <= address < 0x20040000 else hex(address)}
    print(json.dumps({'script': str(OUTPUT), 'device': 'AE822FA0E5597LS0_M55_HE', 'vector_base': '0x80100000',
                      'msp': hex(msp), 'reset': hex(reset), 'symbols': symbols}, indent=2))
    print(f'JLinkExe -device AE822FA0E5597LS0_M55_HE -if SWD -speed 4000 -autoconnect 1 -CommanderScript "{OUTPUT}"')


if __name__ == '__main__':
    main()
