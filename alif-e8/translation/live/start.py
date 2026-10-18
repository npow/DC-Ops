"""Start the debugger-loaded translator with full model-block verification."""
import argparse
import subprocess
import sys
import time
from types import SimpleNamespace
from device import ROOT, FLASHED_ELF, flash, restart, symbols, jlink
from direct_upload import DirectConnection

def loading_status(link, table):
    names = ('live_pipeline_state', 'live_assets_ready', 'live_asset_command',
             'live_integrity_command', 'live_integrity_status')
    state = {name: link.memory_read32(table[name], 1)[0] for name in names}
    vector = link.memory_read32(0xe000ed08, 1)[0]
    reset_vector = link.memory_read32(0x80200004, 1)[0]
    if vector != 0x80200000 or reset_vector != (table['Reset_Handler'] | 1):
        raise RuntimeError('Running firmware does not match the flashed live ELF')
    return state


def require_loading_idle(state):
    if (state['live_pipeline_state'] != 2 or state['live_assets_ready'] != 0 or
            state['live_asset_command'] != 0 or state['live_integrity_command'] != 0 or
            state['live_integrity_status'] == 1):
        raise RuntimeError(f'Firmware is not idle before inference; restart/enable refused: {state}')


def wait_for_asset_loading(table):
    connection = DirectConnection()
    try:
        connection.connect(SimpleNamespace(speed=15000))
        deadline = time.monotonic()+30
        while True:
            state = loading_status(connection.link, table)
            if state['live_pipeline_state'] == 2:
                require_loading_idle(state)
                return
            if (state['live_assets_ready'] or state['live_pipeline_state'] >= 3 or
                    time.monotonic() >= deadline):
                raise RuntimeError(f'Firmware did not reach asset loading state: {state}')
            time.sleep(0.1)
    finally:
        connection.close()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', action='store_true')
parser.add_argument('--pause-ocr', action='store_true', help='Keep camera running and pause OCR for manual diagnostics')
parser.add_argument('--button-capture', action='store_true', help='Use button-triggered still capture instead of automatic live OCR')
args = parser.parse_args()
if args.build:
    subprocess.run(['sh', 'translation/live/build.sh'], cwd=ROOT, check=True)
flash()
table = symbols()
wait_for_asset_loading(table)
for attempt in range(2):
    try:
        subprocess.run([sys.executable, 'translation/live/direct_upload.py', '--elf', str(FLASHED_ELF),
            '--via-staging', '--resume', '--no-enable'], cwd=ROOT, check=True)
        break
    except subprocess.CalledProcessError:
        if attempt:
            raise
        connection = DirectConnection()
        try:
            connection.connect(SimpleNamespace(speed=15000))
            require_loading_idle(loading_status(connection.link, table))
        finally:
            connection.close()
        print('Asset transfer failed while firmware is idle; restarting once and resuming retained assets.', flush=True)
        restart()
        table = symbols()
        wait_for_asset_loading(table)
subprocess.run([sys.executable, 'translation/live/repair_assets.py', '--elf', str(FLASHED_ELF)], cwd=ROOT, check=True)
address = symbols()['live_assets_ready']
commands = []
commands.append(f'w4 0x{table["ocr_single_shot_enabled"]:08x} {int(args.button_capture)}')
if args.pause_ocr:
    commands.append(f'w4 0x{table["live_benchmark_enabled"]:08x} 2')
commands.append(f'w4 0x{address:08x} 1')
jlink(commands, 'live_enable.log')
print('Verified models enabled. Factory boot configuration is preserved.')
