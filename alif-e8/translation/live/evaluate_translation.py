#!/usr/bin/env python3
"""Evaluate frozen translation samples through the running manual service; never flash."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

from device import ARTIFACTS, FLASHED_ELF, NM, symbols
from direct_upload import DirectConnection


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def references():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from prepare import CALIBRATION, CHECKPOINT, EVALUATION, REVISION, load_model
    from translator import Translator

    torch.set_num_threads(4)
    samples_path = ARTIFACTS / 'samples.json'
    samples = json.loads(samples_path.read_text())
    assert samples == {'calibration': CALIBRATION, 'evaluation': EVALUATION}
    assert len(CALIBRATION) == 24 and len(EVALUATION) == 8
    sample_hash = digest(samples_path)
    baseline = json.loads((ARTIFACTS / 'baseline.json').read_text())
    quantized = json.loads((ARTIFACTS / 'spatial_staging/evaluation.json').read_text())
    tosa_path = ARTIFACTS / 'spatial_staging/tosa_evaluation.json'
    tosa = json.loads(tosa_path.read_text())
    assert tosa['samples_sha256'] == sample_hash and tosa['evaluation_denominator'] == 8
    tosa_rows = {row['input']: row for row in tosa['results']+tosa['diagnostics']}
    assert baseline['samples_sha256'] == quantized['samples_sha256'] == sample_hash
    assert baseline['denominator'] == quantized['denominator'] == 8
    assert quantized['checkpoint_revision'] == REVISION
    manifest_path = ARTIFACTS / 'checkpoint_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['revision'] == REVISION
    assert digest(manifest_path) == quantized['checkpoint_manifest_sha256']
    for item in manifest['files']:
        path = CHECKPOINT / item['file']
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], str(path)
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    expected = []
    for index, text in enumerate(EVALUATION + ['T', 'Hello', 'STOP']):
        result = translator.translate(text)
        row = {'input': text, 'group': 'frozen_evaluation' if index < 8 else 'diagnostic',
               'float_text': result['text'], 'float_tokens': result['tokens'],
               'float_eos': result['finished']}
        if index < 8:
            frozen = quantized['results'][index]
            assert frozen['input'] == text and frozen['float_tokens'] == result['tokens']
            row['quantized_host_tokens'] = frozen['staged_tokens']
        if text in tosa_rows:
            row['tosa_reference_tokens'] = tosa_rows[text]['tosa_tokens']
        expected.append(row)
    return expected, {'checkpoint_revision': REVISION, 'samples_sha256': sample_hash,
                      'checkpoint_manifest_sha256': digest(manifest_path),
                      'tosa_reference_sha256': digest(tosa_path),
                      'calibration_denominator': 24, 'frozen_evaluation_denominator': 8,
                      'diagnostic_denominator': 3, 'eos_token_id': model.config.eos_token_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pause-ocr', action='store_true', help='Take ownership of idle mode0 and restore it when safe')
    parser.add_argument('--timeout', type=float, default=60, help='Seconds allowed per phrase (default60)')
    parser.add_argument('--speed', type=int, default=15000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--pause-display', type=int, choices=(0, 1), help='Compare continuous rendering (0) with frozen preview (1)')
    parser.add_argument('--group', choices=('all', 'frozen', 'diagnostic'), default='all')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    expected, audit = references()  # All local checks finish before opening the debugger.
    if args.group != 'all':
        expected = [row for row in expected if row['group'] == ('diagnostic' if args.group == 'diagnostic' else 'frozen_evaluation')]
    directory = args.output or ARTIFACTS / 'live_translation_evaluation' / time.strftime('%Y%m%d-%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(audit, firmware_sha256=digest(FLASHED_ELF), results=[], live_ocr_restored=False,
                  selected_group=args.group, requested_pause_display=args.pause_display)
    table = symbols()
    generated = next(address for name, address in table.items() if name.endswith('9generatedE'))
    busy = next(address for name, address in table.items() if name.endswith('4busyE'))
    sized = subprocess.check_output([NM, '-S', '-n', str(FLASHED_ELF)], text=True)
    stats = next((int(parts[0], 16) for line in sized.splitlines()
                  if len(parts := line.split()) == 4 and parts[1] == '00000060'
                  and parts[3].endswith('5statsE')), None)
    connection = DirectConnection()
    owned = False
    pending = False
    old_display_policy = None
    link = None

    def read(name):
        return link.memory_read32(table[name], 1)[0]

    def write(name, value):
        if link.memory_write32(table[name], [value]) != 4:
            raise RuntimeError(f'Failed write: {name}')

    def validate_live():
        if link.memory_read32(0xe000ed08, 1)[0] != 0x80200000:
            raise RuntimeError('Live firmware vector base changed')
        if read('live_assets_ready') != 1 or read('live_pipeline_state') != 4:
            raise RuntimeError('Live assets are not initialized')

    def inactive():
        return (not read('manual_translation_command') and read('manual_translation_status') != 1
                and not read('live_benchmark_command')
                and read('live_benchmark_status') not in (1, 2, 3, 4)
                and not read('live_asset_command') and not read('live_integrity_command')
                and read('live_integrity_status') != 1)

    def worker_idle():
        request = link.memory_read32(0x02680008, 1)[0]
        response = link.memory_read32(0x02680080, 1)[0]
        return not request or request == response

    def snapshot():
        fields = ('live_frames', 'live_milliseconds', 'translator_lcd_presents',
                  'translator_lcd_timeouts', 'translator_lcd_errors', 'live_display_paused',
                  'live_display_pause_count', 'live_display_last_pause_ms', 'live_display_total_pause_ms')
        fields = [name for name in fields if name in table]
        result = {name: read(name) for name in fields}
        if stats is not None:
            words = link.memory_read32(stats, 24)
            raw = struct.pack('<24I', *words)
            keys = ('invocations', 'failures', 'elapsed_ms', 'last_ms', 'total_cycles',
                    'active_cycles', 'mac_cycles', 'idle_cycles')
            result['npu'] = {name: dict(zip(keys, struct.unpack_from('<IIIIQQQQ', raw, index*48)))
                             for index, name in enumerate(('HP_U55', 'HG_U85'))}
        return result

    def save():
        rows = report['results']
        report['completed'] = len(rows)
        for group, denominator in (('frozen_evaluation', 8), ('diagnostic', 3)):
            subset = [row for row in rows if row['group'] == group]
            report[group] = {'denominator': denominator, 'completed': len(subset),
                             'float_token_matches': sum(row['matches_float_tokens'] for row in subset),
                             'eos_finished': sum(row['eos'] for row in subset),
                             'successful': sum(row['status'] == 2 and row['error'] == 0 for row in subset)}
            compared = [row for row in subset if 'matches_tosa_reference_tokens' in row]
            report[group]['tosa_reference_denominator'] = len(compared)
            report[group]['tosa_reference_matches'] = sum(row['matches_tosa_reference_tokens'] for row in compared)
        (directory / 'result.json').write_text(json.dumps(report, indent=2)+'\n')

    try:
        connection.connect(SimpleNamespace(speed=args.speed))
        link = connection.link
        validate_live()
        if not inactive():
            raise RuntimeError('Another diagnostic owns inference')
        mode = read('live_benchmark_enabled')
        if mode == 0:
            if not args.pause_ocr:
                raise RuntimeError('Use --pause-ocr to own and pause live OCR')
            write('live_benchmark_enabled', 2)
            owned = True
        elif mode != 2:
            raise RuntimeError(f'Inference mode{mode} belongs to another diagnostic')
        deadline = time.monotonic()+args.timeout
        while link.memory_read8(busy, 1)[0] or not worker_idle():
            validate_live()
            if time.monotonic() >= deadline:
                raise TimeoutError('OCR did not relinquish shared buffers')
            time.sleep(.1)
        frame = read('live_frames')
        deadline = time.monotonic()+5
        while ((read('live_frames')-frame) & 0xffffffff) < 2:
            if time.monotonic() >= deadline:
                raise TimeoutError('Main loop did not resume camera frames')
            time.sleep(.02)
        report['before'] = snapshot()
        if args.pause_display is not None:
            if 'manual_translation_pause_display' not in table:
                raise RuntimeError('Flashed firmware lacks display-pause selection')
            old_display_policy = read('manual_translation_pause_display')
            write('manual_translation_pause_display', args.pause_display)
        for reference in expected:
            validate_live()
            if not inactive() or link.memory_read8(busy, 1)[0] or not worker_idle():
                raise RuntimeError('Shared inference buffers are not idle')
            raw = reference['input'].encode()+b'\0'
            if len(raw) > 256:
                raise RuntimeError('Manual input exceeds256 bytes')
            connection.write_bytes(table['manual_translation_input'], raw.ljust(256, b'\0'), time.monotonic()+5)
            before = snapshot()
            write('manual_translation_status', 0)
            started = time.monotonic()
            write('manual_translation_command', 1)
            pending = True
            while read('manual_translation_command') or read('manual_translation_status') not in (2, 0xe0):
                validate_live()
                if time.monotonic()-started >= args.timeout:
                    raise TimeoutError(f"Translation timed out: {reference['input']}; inference remains paused")
                time.sleep(.05)
            pending = False
            count = read('tr_text_tokens')
            if count > 31:
                raise RuntimeError(f'Invalid generated token count{count}')
            tokens = link.memory_read32(generated, count) if count else []
            text = bytes(link.memory_read8(table['manual_translation_output'], 512)).split(b'\0', 1)[0].decode('utf-8', errors='replace')
            error = read('manual_translation_error')
            if error >= 0x80000000:
                error -= 0x100000000
            row = dict(reference, status=read('manual_translation_status'), error=error,
                       text=text, tokens=tokens, token_count=count,
                       eos=bool(tokens and tokens[-1] == audit['eos_token_id']),
                       truncated=bool(read('tr_text_truncated')),
                       elapsed_ms=read('manual_translation_elapsed_ms'),
                       host_wall_seconds=time.monotonic()-started,
                       matches_float_tokens=tokens == reference['float_tokens'],
                       before=before, after=snapshot())
            if 'quantized_host_tokens' in reference:
                row['matches_quantized_host_tokens'] = tokens == reference['quantized_host_tokens']
            if 'tosa_reference_tokens' in reference:
                row['matches_tosa_reference_tokens'] = tokens == reference['tosa_reference_tokens']
            report['results'].append(row)
            save()
            print(json.dumps({key: row[key] for key in ('input', 'text', 'tokens', 'error', 'eos', 'matches_float_tokens', 'elapsed_ms')}), flush=True)
            if error not in (0, -26):
                pending = True
                raise RuntimeError(f"Execution failed for {reference['input']}: {error}; inference remains paused")
        report['after'] = snapshot()
    except Exception as error:
        report['failure'] = str(error)
        raise
    finally:
        try:
            if old_display_policy is not None and link is not None and not pending and inactive():
                write('manual_translation_pause_display', old_display_policy)
            if owned and link is not None:
                validate_live()
                if pending or not inactive() or link.memory_read8(busy, 1)[0] or not worker_idle():
                    report['restore_blocked'] = 'An executing job may still own buffers; mode2 retained'
                elif read('live_benchmark_enabled') != 2:
                    report['restore_blocked'] = 'Inference mode ownership changed'
                else:
                    write('live_benchmark_enabled', 0)
                    report['live_ocr_restored'] = True
        finally:
            save()
            connection.close()
    print(f'Results: {directory / "result.json"}', flush=True)


if __name__ == '__main__':
    main()
