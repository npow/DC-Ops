"""Flash and inspect the live translator without changing factory boot configuration."""
import argparse
import json
import re
import struct
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/translator'
ELF = ROOT / 'out/translator-live/DevKit-E8/Debug/translator-live.elf'
FLASHED_ELF = ARTIFACTS / 'flashed_live.elf'
NM = '/Users/npow/.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin/arm-none-eabi-nm'

def symbols(elf=None):
    image = elf or (FLASHED_ELF if FLASHED_ELF.exists() else ELF)
    output = subprocess.check_output([NM, '-n', str(image)], text=True)
    return {p[2]: int(p[0], 16) for line in output.splitlines() if len(p := line.split()) == 3}

def jlink(commands, name, timeout=120, device='AE822FA0E5597LS0_M55_HP'):
    result = subprocess.run(['JLinkExe', '-device', device, '-if', 'SWD', '-speed', '15000',
        '-autoconnect', '1'], input='\n'.join([*commands, 'exit', '']), capture_output=True,
        text=True, timeout=timeout)
    log = result.stdout + result.stderr
    (ARTIFACTS / name).write_text(log)
    if result.returncode or re.search(r'Cannot connect|Error while|Could not (?:read|write)|Failed to download|CPU could not be halted|Could not find core', log):
        raise RuntimeError(log[-4000:])
    return log

def flash():
    table = symbols(ELF)
    clear = [f'w4 0x{0xe0002008+i*4:08x} 0' for i in range(8)]
    jlink(['h', f'loadfile {ELF}', 'r', 'w4 0xe000ed08 0x80200000',
        'wreg MSP 0x20100000', f'SetPC 0x{table["Reset_Handler"]:08x}', *clear, 'g'], 'live_flash.log')
    shutil.copy2(ELF, FLASHED_ELF)
    print('Live translator started; factory boot configuration preserved.')

def restart():
    table = symbols()
    clear = [f'w4 0x{0xe0002008+i*4:08x} 0' for i in range(8)]
    jlink(['h', 'r', 'w4 0xe000ed08 0x80200000', 'wreg MSP 0x20100000',
           f'SetPC 0x{table["Reset_Handler"]:08x}', *clear, 'g'], 'live_restart.log')
    print('Live firmware restarted; external RAM will be reinitialized without reflashing.')

def status():
    table = symbols()
    names = ['live_status', 'live_error', 'live_frames', 'live_milliseconds',
        'live_pipeline_state', 'live_pipeline_error', 'live_assets_ready',
        'live_he_boot_stage', 'live_he_boot_transport', 'live_he_boot_service', 'live_he_seen_status',
        'translator_lcd_presents', 'translator_lcd_timeouts', 'translator_lcd_errors',
        'ocr_state', 'ocr_runs', 'ocr_error', 'ocr_boxes', 'ocr_confidence_x1000',
        'ocr_detector_ms', 'ocr_recognizer_ms', 'ocr_translator_ms',
        'ocr_stage_copy_ms', 'ocr_stage_service_ms', 'ocr_stable_matches',
        'ocr_stable_elapsed_ms', 'ocr_cache_hits', 'ocr_pause_display_during_translation',
        'live_display_paused', 'live_display_pause_count', 'live_display_last_pause_ms',
        'live_display_total_pause_ms',
        'live_button_down', 'live_button_presses', 'ocr_capture_command', 'ocr_capture_state',
        'live_button_pins', 'live_button_events',
        'ocr_single_shot_enabled',
        'live_show_ocr_regions',
        'ocr_sign_mode', 'ocr_sign_status', 'ocr_sign_lookup_ms', 'ocr_capture_elapsed_ms',
        'tr_text_state', 'tr_text_stage', 'tr_text_tokens', 'tr_text_error', 'tr_text_truncated',
        'tr_stage_program_address', 'tr_stage_copy_ms', 'tr_stage_service_ms',
        'tr_stage_prepare_ms', 'tr_stage_execute_ms',
        'manual_translation_command','manual_translation_status','manual_translation_error','manual_translation_elapsed_ms']
    names = [n for n in names if n in table]
    commands = ['mem32 0xe000ed08 1']
    for name in names:
        commands.append(f'savebin {ARTIFACTS / (name+".live.bin")} 0x{table[name]:08x} 4')
    for name, size in [('dc_log', 16384), ('dc_log_position', 4), ('ocr_english', 256), ('ocr_spanish', 512), ('tr_text_english', 512), ('tr_text_spanish', 512), ('manual_translation_input',256), ('manual_translation_output',512)]:
        if name in table:
            commands.append(f'savebin {ARTIFACTS / (name+".live.bin")} 0x{table[name]:08x} 0x{size:x}')
    sized = subprocess.check_output([NM, '-S', '-n', str(FLASHED_ELF)], text=True)
    npu_address = next((int(parts[0],16) for line in sized.splitlines()
        if len(parts := line.split()) == 4 and parts[1] == '00000060' and parts[3].endswith('5statsE')), None)
    if npu_address is not None:
        commands.append(f'savebin {ARTIFACTS / "npu_samples.live.bin"} 0x{npu_address:08x} 0x60')
    if any('he_measurementE' in name for name in table):
        he_address = next(address for name,address in table.items() if 'he_measurementE' in name)
        commands.append(f'savebin {ARTIFACTS / "he_measurement.live.bin"} 0x{he_address:08x} 0x80')
    commands += []
    log = jlink(commands, 'live_status.log')
    vector = re.search(r'E000ED08 = ([0-9A-Fa-f]+)', log)
    if not vector or int(vector.group(1),16) != 0x80200000:
        raise RuntimeError('Live translator is not running: vector base does not match.')
    result = {n: struct.unpack('<I', (ARTIFACTS / (n+'.live.bin')).read_bytes())[0] for n in names}
    for n in ('ocr_english', 'ocr_spanish', 'tr_text_english', 'tr_text_spanish', 'manual_translation_input', 'manual_translation_output'):
        if n in table:
            result[n] = (ARTIFACTS / (n+'.live.bin')).read_bytes().split(b'\0')[0].decode(errors='replace')
    if npu_address is not None:
        samples = (ARTIFACTS / 'npu_samples.live.bin').read_bytes()
        keys = ['invocations','failures','elapsed_ms','last_ms','total_cycles','active_cycles','mac_cycles','idle_cycles']
        result['npu'] = {name: dict(zip(keys,struct.unpack_from('<IIIIQQQQ',samples,index*48)))
            for index,name in enumerate(['HP_U55','HG_U85'])}
    if (ARTIFACTS / 'he_measurement.live.bin').exists():
        sample = struct.unpack_from('<IiIIQIII',(ARTIFACTS / 'he_measurement.live.bin').read_bytes())
        result['HE_U55_last'] = dict(zip(['sequence','error','output_bytes','elapsed_us','total_cycles','active_cycles','mac_cycles','idle_cycles'],sample))
    raw = (ARTIFACTS / 'dc_log.live.bin').read_bytes()
    pos = struct.unpack('<I', (ARTIFACTS / 'dc_log_position.live.bin').read_bytes())[0]
    raw = raw[:pos] if pos < len(raw) else raw[pos % len(raw):]+raw[:pos % len(raw)]
    print(raw.decode(errors='replace')[-1800:])
    print(json.dumps(result, indent=2))
    (ARTIFACTS / 'live_status.json').write_text(json.dumps(result, indent=2)+'\n')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['flash', 'status'])
    args = parser.parse_args()
    flash() if args.action == 'flash' else status()
