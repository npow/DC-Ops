"""Exercise the capture event on the E8; does not prove physical button wiring or OCR accuracy."""
import hashlib
import json
import time
from types import SimpleNamespace
from device import ARTIFACTS, FLASHED_ELF, symbols
from direct_upload import DirectConnection


table = symbols()
names = ('live_pipeline_state', 'live_error', 'live_frames', 'live_display_paused',
         'translator_lcd_errors', 'translator_lcd_timeouts', 'ocr_runs', 'ocr_error',
         'ocr_capture_state', 'ocr_capture_requests', 'ocr_captured_frame',
         'ocr_single_shot_enabled', 'ocr_detector_ms', 'ocr_recognizer_ms',
         'ocr_translator_ms', 'ocr_boxes', 'live_button_presses',
         'ocr_sign_mode', 'ocr_sign_status', 'ocr_sign_lookup_ms', 'ocr_capture_elapsed_ms',
         'tr_text_tokens', 'tr_text_state')
connection = DirectConnection()
report = {'event_source': 'debug command invoking the same handler as button polling',
          'physical_button_verified': False, 'ocr_accuracy_verified': False,
          'firmware_sha256': hashlib.sha256(FLASHED_ELF.read_bytes()).hexdigest()}
try:
    connection.connect(SimpleNamespace(speed=15000))
    link = connection.link

    def state():
        return {name: link.memory_read32(table[name], 1)[0] for name in names}

    before = state()
    assert before['live_pipeline_state'] == 4 and before['ocr_single_shot_enabled'] == 1, before
    assert before['ocr_capture_state'] == 0 and not before['live_display_paused'], before
    time.sleep(1)
    idle = state()
    assert idle['live_frames'] > before['live_frames'] and idle['ocr_runs'] == before['ocr_runs'], idle
    link.memory_write32(table['ocr_capture_command'], [1])
    deadline = time.monotonic() + 30
    while True:
        captured = state()
        if captured['ocr_capture_state'] == 3:
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f'Capture timed out: {captured}')
        time.sleep(0.1)
    assert captured['live_display_paused'] == 1, captured
    assert captured['ocr_runs'] == idle['ocr_runs'] + 1 and not captured['ocr_error'], captured
    assert captured['live_frames'] == captured['ocr_captured_frame'], captured
    if captured['ocr_sign_mode']:
        assert captured['ocr_translator_ms'] < 5, captured
        assert captured['tr_text_tokens'] == before['tr_text_tokens'], captured
        assert captured['tr_text_state'] == before['tr_text_state'], captured
    for name, size in (('ocr_english', 256), ('ocr_spanish', 512)):
        report[name] = bytes(link.memory_read8(table[name], size)).split(b'\0')[0].decode(errors='replace')
    time.sleep(1)
    held = state()
    assert held['live_frames'] == captured['live_frames'] and held['ocr_runs'] == captured['ocr_runs'], held
    link.memory_write32(table['ocr_capture_command'], [1])
    deadline = time.monotonic() + 5
    while True:
        resumed = state()
        if resumed['ocr_capture_state'] == 0 and resumed['live_frames'] > held['live_frames']:
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f'Resume timed out: {resumed}')
        time.sleep(0.1)
    time.sleep(1)
    after = state()
    assert not after['live_display_paused'] and after['ocr_runs'] == captured['ocr_runs'], after
    for name in ('live_error', 'translator_lcd_errors', 'translator_lcd_timeouts'):
        assert after[name] == 0, after
    report.update(passed=True, before=before, idle=idle, captured=captured, held=held, resumed=resumed, after=after)
finally:
    connection.close()
    path = ARTIFACTS / 'button_capture_check.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(path)
