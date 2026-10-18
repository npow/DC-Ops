"""Preserve completed timings when USB disconnects before final readback."""
import ast,datetime,json,re,statistics,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
log=root/'board_fixed_latency_allocator_corrected.log'
rows=re.findall(r'state/error/run/phase (\([^\n]+?\)) total_ms (\[[^\n]+?\])',log.read_text())
completed=[]
for state,values in rows:
 s=ast.literal_eval(state);v=ast.literal_eval(values)
 if s[0]==4:
  # Entering run N means all preceding chunks passed finite-output checks.
  completed=v[:s[2]]
measured=completed[1:]
r=json.loads((root/'metadata.json').read_text())
r.update(status='completed_predictions_usb_disconnected_before_final_readback',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),board='Alif E8 M55 HP + Ethos-U85-256',quantization='INT8 weights and activations',completed_predictions=len(completed),warmup_runs=1,measured_runs=len(measured),completed_total_ms=completed,measured_total_ms=measured,median_measured_ms=statistics.median(measured),range_measured_ms=[min(measured),max(measured)],all_completed_outputs_finite=True,final_action_values_captured=False,stage_timings_captured=False,psram_bus_hz=25000000,psram_read_latency='Fixed MR0bit5 both dies;8 XIPcycles;RXDSvariablelatencyoff',method_metadata='Internal SRAM512KiB per model',external_axi='One outstanding read/write;64-byte bursts',command_stream='Original Vela commands copied into internal SRAM;no serialization or fragmentation',timing_scope='Two480x640camera encoder passes+transformer+feature copies. Synthetic normalized float input reused for both cameras. Excludes capture/preprocessing,upload/load,and physical robot execution.',quality_scope='One synthetic calibration/input; no policy-quality validation or board-vs-host numerical comparison.',timer_check=json.loads((root/'timer_check.json').read_text()),transfer_audit=json.loads((root/'upload_audit.json').read_text()),evidence_log=log.name,evidence_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
(root/'board_partial_report.json').write_text(json.dumps(r,indent=2)+'\n')
print({k:r[k] for k in ['status','completed_predictions','measured_runs','measured_total_ms','median_measured_ms']})
s=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {};s.update(status=r['status'],measured_board_runs=len(measured),completed_board_predictions=len(completed),measured_median_ms=r['median_measured_ms'],board_partial_report='board_partial_report.json',blocking_condition='J-Link USB disappeared duringfifthmeasuredrepeat;4measuredchunks+1warmup completed. Need reconnect withoutreset toharvestfinalmailbox.',next_action='Reconnect board and read mailbox withoutreset; then handofftoOcto with explicitrestorationownership.')
(root/'status.json').write_text(json.dumps(s,indent=2)+'\n')
