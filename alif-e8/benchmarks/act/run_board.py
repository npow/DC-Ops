"""Load a latency-only ACT harness. Uses no robot/motor interfaces."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, '/private/tmp/dcops-pylink')
import pylink

ROOT = Path(__file__).resolve().parent
TOOLS = Path(os.environ.get('ARM_GCC_BIN', Path.home()/'.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin'))
ELF = ROOT/'firmware/out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.elf'
LIB = os.environ.get('JLINK_LIBRARY', '/Applications/SEGGER/JLink_V986a/libjlinkarm.9.86.1.dylib')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--flash',action='store_true')
    parser.add_argument('--upload',action='store_true')
    parser.add_argument('--reuse-upload',action='store_true')
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--connect-speed',type=int,default=15000)
    parser.add_argument('--monitor-speed',type=int,default=1000)
    args=parser.parse_args()
    lock=open('/tmp/alif-e8-1219599493.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    symbols={line.split()[2]:int(line.split()[0],16)
        for line in subprocess.check_output([str(TOOLS/'arm-none-eabi-nm'),'-n',str(ELF)],text=True).splitlines()
        if len(line.split())==3 and line.split()[0]!='U'}
    mailbox=symbols['act_benchmark']
    link=pylink.JLink(lib=pylink.library.Library(dllpath=LIB))
    link.open(serial_no=1219599493)
    link.set_tif(pylink.enums.JLinkInterfaces.SWD)
    link.connect('AE822FA0E5597LS0_M55_HP',speed=args.connect_speed)
    link.exec_command('SetEnableFlashbank 0xA0000000=0')
    link.exec_command('SetAllowFlashCache=0')
    read=link._dll.JLINKARM_ReadMemEx
    read.argtypes=[ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32]
    read.restype=ctypes.c_int
    write=link._dll.JLINKARM_WriteMemEx
    write.argtypes=read.argtypes; write.restype=ctypes.c_int

    def get_bytes(address,size):
        buf=ctypes.create_string_buffer(size)
        n=read(address,size,buf,4)
        if n!=size: raise RuntimeError(f'read {address:#x}: {n}/{size}')
        return buf.raw

    def staging_job(address,size,operation):
        control=mailbox+128+1200
        sequence=link.memory_read32(control+16,1)[0]+1
        link.memory_write32(control,[address,size])
        link.memory_write32(control+20,[operation])
        link.memory_write32(control+12,[sequence])
        deadline=time.monotonic()+30
        while link.memory_read32(control+16,1)[0]!=sequence:
            if time.monotonic()>deadline: raise TimeoutError('SRAM staging job timed out')
            time.sleep(.001)
        return link.memory_read32(control+8,1)[0]

    def put_file(path,address):
        raw=path.read_bytes();digest=hashlib.sha256()
        if args.reuse_upload:
            actual_crc=staging_job(address,len(raw),2)
            if actual_crc==zlib.crc32(raw):
                result={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                    'all_bytes_verified':True,'cpu_crc32':actual_crc,'retained':True,
                    'verification':'full-file on-device CRC32 against source; source SHA256 recorded'}
                print('retained',path.name,result,flush=True)
                return result
        for offset in range(0,len(raw),64*1024):
            block=raw[offset:offset+64*1024]
            padded=block+bytes((-len(block))%4)
            buffer=ctypes.create_string_buffer(padded)
            count=write(symbols['temp_pool'],len(padded),buffer,4)
            if count!=len(padded):raise RuntimeError(f'write {address+offset:#x}: {count}/{len(padded)}')
            actual_crc=staging_job(address+offset,len(padded),1)
            if actual_crc!=zlib.crc32(padded):raise RuntimeError(f'upload CRC mismatch {path.name} offset {offset}')
            digest.update(block)
            if offset%(1024*1024)==0: print('upload+verify',path.name,offset,'/',len(raw),flush=True)
        actual_crc=staging_job(address,len(raw),2)
        if actual_crc!=zlib.crc32(raw): raise RuntimeError(f'full-file CRC mismatch {path.name}')
        result={'bytes':len(raw),'sha256':digest.hexdigest(),'all_bytes_verified':True,'cpu_crc32':actual_crc,'verification':'per-block + full-file on-device CRC32 against source; source SHA256 recorded'}
        print('uploaded',path.name,result,flush=True)
        return result

    try:
        if args.flash:
            link.halt()
            backup=ROOT/'pre_act_hp_mram.bin'
            if not backup.exists():
                with backup.open('wb') as out:
                    for offset in range(0,0x370000,64*1024):
                        out.write(get_bytes(0x80200000+offset,min(64*1024,0x370000-offset)))
                print('previous HP firmware backed up',flush=True)
            link.flash_file(str(ELF),0)
            link.reset(halt=True)
            vector=get_bytes(0x80200000,8)
            sp,pc=struct.unpack('<II',vector)
            link.memory_write32(0xe000ed08,[0x80200000])
            registers={link.register_name(i):i for i in link.register_list()}
            link.register_write(registers['MSP'],sp)
            pc_index=next((index for name,index in registers.items() if name=='PC' or name.startswith('R15')),None)
            if pc_index is None: raise RuntimeError(f'No program counter register in {registers}')
            link.register_write(pc_index,pc)
            link.restart()
            time.sleep(1)
        words=link.memory_read32(mailbox,32)
        print('initial mailbox',words[:8],flush=True)
        if words[0]!=0x41435438:raise RuntimeError('ACT firmware is not running')
        tick_address=symbols['act_ticks']
        t0=time.monotonic();tick0=link.memory_read32(tick_address,1)[0]
        time.sleep(2)
        tick1=link.memory_read32(tick_address,1)[0];elapsed=time.monotonic()-t0
        timer={'host_elapsed_seconds':elapsed,'device_elapsed_ms':tick1-tick0,'ratio':(tick1-tick0)/(1000*elapsed)}
        (ROOT/'timer_check.json').write_text(json.dumps(timer,indent=2)+'\n')
        if not .98<timer['ratio']<1.02:raise RuntimeError(f'timer verification failed {timer}')
        artifacts={}
        if not args.upload and (ROOT/'upload_audit.json').exists():
            artifacts=json.loads((ROOT/'upload_audit.json').read_text())
        if args.upload:
            if words[1]!=2 or words[2]:raise RuntimeError('firmware is not waiting for upload')
            link.set_speed(15000)
            assert (ROOT/'vision.pte').stat().st_size<12*1024*1024
            assert (ROOT/'head.pte').stat().st_size<24*1024*1024
            for name,address in [('vision.pte',0xa0000000),('head.pte',0xa0c00000),('image.bin',0xa2400000)]:
                artifacts[name]=put_file(ROOT/name,address)
            (ROOT/'upload_audit.json').write_text(json.dumps(artifacts,indent=2)+'\n')
            link.set_speed(args.monitor_speed)
            link.memory_write32(mailbox+12,[(ROOT/'vision.pte').stat().st_size,(ROOT/'head.pte').stat().st_size])
            link.memory_write32(mailbox+8,[1])
        else:
            link.set_speed(args.monitor_speed)
        deadline=time.monotonic()+args.timeout;previous=None
        while time.monotonic()<deadline:
            words=link.memory_read32(mailbox,32)
            status=(words[1],words[5],words[6],words[7])
            if status!=previous:
                print('state/error/run/phase',status,'total_ms',words[26:32],flush=True);previous=status
            if words[1]>=90:
                message=get_bytes(symbols['act_last_error'],512).split(b'\0',1)[0].decode(errors='replace')
                print('runtime error',message,flush=True)
                snapshot=link.memory_read32(symbols['act_npu_snapshot'],163)
                failure={'words':words,'runtime_message':message,'npu_snapshot':snapshot}
                command_bytes=link.memory_read32(symbols['act_command_bytes'],1)[0]
                if 0 < command_bytes <= 1024*1024:
                    commands=get_bytes(symbols['temp_pool'],command_bytes)
                    (ROOT/'failed_sram_commands.bin').write_bytes(commands)
                    failure['sram_commands']={'bytes':command_bytes,'sha256':hashlib.sha256(commands).hexdigest()}
                    # The driver remap may copy a short tail beyond the command
                    # payload. Compare only the canonical stream from Vela.
                    import numpy as np
                    stage='vision' if words[7] in (3,4) else 'head'
                    expected=np.load(ROOT/f'{stage}_vela/output/out_vela.npz')['cmd_data'].tobytes()[32:]
                    failure['sram_commands']['canonical_stream_matches']=commands[:len(expected)]==expected
                    failure['sram_commands']['canonical_bytes']=len(expected)
                (ROOT/'board_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                raise RuntimeError(f'board error {status}')
            if words[1]==5:
                actions=struct.unpack('<300f',get_bytes(mailbox+128,1200))
                report=json.loads((ROOT/'metadata.json').read_text())
                report.update({'status':'completed_board_latency_smoke','board':'Alif E8 M55 HP + Ethos-U85-256',
                    'core_clock_hz':link.memory_read32(symbols['SystemCoreClock'],1)[0],
                    'psram_bus_hz':25000000,'warmup_runs':1,'measured_runs':5,
                    'debug_monitor_khz':args.monitor_speed,
                    'command_stream_location':'Internal SRAM staging pool; copy included in inference timing',
                    'external_axi':'One outstanding read, one outstanding write, 64-byte maximum burst',
                    'timer_check':timer,'runtime':'ExecuTorch 1.5.1','quantization':'INT8 weights and activations; CPU float input/output conversion',
                    'front_ms':words[8:14],'side_ms':words[14:20],'head_ms':words[20:26],'total_ms':words[26:32],
                    'warm_median_ms':statistics.median(words[27:32]),'artifacts':artifacts,
                    'actions':actions,'all_outputs_finite':all(__import__('math').isfinite(x) for x in actions),
                    'timing_scope':'Two camera encoder executions + transformer + feature copies. Float normalized synthetic images, same image reused for both cameras. Excludes image capture/preprocessing, upload/load and robot execution. One synthetic calibration sample: no task-quality claim.'})
                (ROOT/'board_report.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({k:report[k] for k in ['status','front_ms','side_ms','head_ms','total_ms','warm_median_ms','core_clock_hz']},indent=2),flush=True)
                return
            time.sleep(1)
        raise TimeoutError('benchmark deadline exceeded')
    finally:
        link.close()
        fcntl.flock(lock,fcntl.LOCK_UN)


if __name__=='__main__':main()
