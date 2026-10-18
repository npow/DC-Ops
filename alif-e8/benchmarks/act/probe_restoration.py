"""Boot preserved original firmware and verify one 64KiB staged PSRAM transfer."""
import ctypes
import fcntl
import hashlib
import json
import struct
import subprocess
import sys
import time
from restore_original import ROOT, APP, NM, connection


def main():
    elf=APP/'artifacts/translator/flashed_live.elf'
    table={line.split()[2]:int(line.split()[0],16)
        for line in subprocess.check_output([NM,'-n',str(elf)],text=True).splitlines()
        if len(line.split())==3 and line.split()[0]!='U'}
    sys.path.insert(0,str(APP/'translation/live'))
    from load_assets import ASSETS,ARTIFACTS
    name,destination,_=ASSETS[0]
    raw=(ARTIFACTS/name).read_bytes()[:65536]
    with open('/tmp/alif-e8-1219599493.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        link=connection()
        try:
            link.halt()
            link.flash_file(str(elf),0)
            link.reset(halt=True)
            sp,pc=link.memory_read32(0x80200000,2)
            link.memory_write32(0xe000ed08,[0x80200000])
            regs={link.register_name(i):i for i in link.register_list()}
            link.register_write(regs['MSP'],sp)
            index=next(v for k,v in regs.items() if k=='PC' or k.startswith('R15'))
            link.register_write(index,pc)
            link.restart()
            time.sleep(2)
            state=link.memory_read32(table['live_pipeline_state'],1)[0]
            ready=link.memory_read32(table['live_assets_ready'],1)[0]
            if state!=2 or ready:raise RuntimeError(f'Original app not idle: state={state}, ready={ready}')
            link.set_speed(4000)
            write=link._dll.JLINKARM_WriteMemEx
            write.argtypes=[ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32]
            write.restype=ctypes.c_int
            buf=ctypes.create_string_buffer(raw)
            staging=table['live_model_staging']
            if write(staging,len(raw),buf,4)!=len(raw):raise RuntimeError('Short SRAM upload')
            source_read=struct.pack('<16384I',*link.memory_read32(staging,16384))
            if source_read!=raw:raise RuntimeError('SRAM staging readback differs')
            sequence=link.memory_read32(table['live_asset_completed_sequence'],1)[0]+1
            for key,value in [('destination',destination),('bytes',len(raw)),('sequence',sequence),('command',1)]:
                link.memory_write32(table['live_asset_'+key],[value])
            deadline=time.monotonic()+15
            while link.memory_read32(table['live_asset_completed_sequence'],1)[0]!=sequence:
                if time.monotonic()>deadline:raise TimeoutError('Original staging service timeout')
                time.sleep(.05)
            error=link.memory_read32(table['live_asset_error'],1)[0]
            actual=struct.pack('<16384I',*link.memory_read32(destination,16384))
            report={'bytes':len(raw),'source_sha256':hashlib.sha256(raw).hexdigest(),
                'readback_sha256':hashlib.sha256(actual).hexdigest(),'mismatching_bytes':sum(a!=b for a,b in zip(raw,actual)),
                'service_error':error,'source_sram_verified':True,'passed':actual==raw and error==0}
            (ROOT/'restoration_probe_after_power_cycle.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2),flush=True)
            if not report['passed']:raise RuntimeError('PSRAM probe still differs after power cycle')
        finally:
            link.close()


if __name__=='__main__':main()
