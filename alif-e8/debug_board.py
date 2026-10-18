"""Run a bounded GDB session against the E8's on-board J-Link."""
import socket
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
with (ROOT / 'artifacts/gdb_server.log').open('w') as log:
    server = subprocess.Popen(['JLinkGDBServer', '-device', 'AE822FA0E5597LS0_M55_HP',
        '-if', 'SWD', '-speed', '4000', '-port', '2337', '-swoport', '2338',
        '-telnetport', '2339', '-nogui'], stdout=log, stderr=log)
    try:
        for _ in range(100):
            if 'Waiting for GDB' in (ROOT / 'artifacts/gdb_server.log').read_text(): break
            if server.poll() is not None: raise RuntimeError('GDB server stopped')
            time.sleep(.1)
        result = subprocess.run([
            '/Users/npow/.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin/arm-none-eabi-gdb',
            '-q', '-batch', str(ROOT / 'out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.elf'),
            '-ex', 'target remote localhost:2337', '-x', sys.argv[1],
            '-ex', 'delete breakpoints', '-ex', 'detach'],
            capture_output=True, text=True, timeout=40)
        print(result.stdout, result.stderr)
    finally:
        server.terminate()
        server.wait(timeout=5)
        cleanup = 'h\n' + ''.join(f'w4 0x{0xe0002008+i*4:08x} 0\n' for i in range(8)) + 'g\nexit\n'
        result = subprocess.run(['JLinkExe', '-device', 'AE822FA0E5597LS0_M55_HP',
            '-if', 'SWD', '-speed', '4000', '-autoconnect', '1'], input=cleanup,
            capture_output=True, text=True, timeout=15)
        (ROOT / 'artifacts/gdb_cleanup.log').write_text(result.stdout + result.stderr)
