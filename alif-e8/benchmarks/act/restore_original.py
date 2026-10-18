"""Restore the pre-benchmark live app and its verified PSRAM model assets."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import argparse

sys.path.insert(0,'/private/tmp/dcops-pylink')
import pylink

ROOT=Path(__file__).resolve().parent
APP=Path(os.environ.get('ALIF_APP_ROOT', ROOT.parents[1]))
LIB=os.environ.get('JLINK_LIBRARY','/Applications/SEGGER/JLink_V986a/libjlinkarm.9.86.1.dylib')
NM=str(Path(os.environ.get('ARM_GCC_BIN',Path.home()/'.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin'))/'arm-none-eabi-nm')


def connection():
    link=pylink.JLink(lib=pylink.library.Library(dllpath=LIB))
    link.open(serial_no=1219599493)
    link.set_tif(pylink.enums.JLinkInterfaces.SWD)
    link.connect('AE822FA0E5597LS0_M55_HP',speed=1000)
    return link


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-assets',action='store_true')
    args=parser.parse_args()
    with open('/tmp/alif-e8-1219599493.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        python=str(APP/'translation/.venv/bin/python')
        if args.resume_assets:
            result=subprocess.run([python,'translation/live/direct_upload.py','--elf',str(APP/'artifacts/translator/flashed_live.elf'),
                '--via-staging','--resume','--no-enable','--speed','4000','--chunk-bytes','0x10000'],cwd=APP)
            if result.returncode:raise RuntimeError('Resumed original asset upload failed')
            subprocess.run([python,'translation/live/repair_assets.py','--speed','4000'],cwd=APP,check=True)
            sys.path.insert(0,str(APP/'translation/live'))
            from device import symbols,jlink
            table=symbols()
            jlink([f'w4 0x{table["live_assets_ready"]:08x} 1'],'act_restore_enable.log')
        else:
            link=connection()
            try:
                link.reset(halt=True)
            finally:
                link.close()
            result=subprocess.run([python,'translation/live/start.py'],cwd=APP)
        if result.returncode:
            raise RuntimeError(f'Original app/assets restoration failed: {result.returncode}')
        elf=APP/'artifacts/translator/flashed_live.elf'
        symbols={line.split()[2]:int(line.split()[0],16)
            for line in subprocess.check_output([NM,'-n',str(elf)],text=True).splitlines()
            if len(line.split())==3 and line.split()[0]!='U'}
        link=connection()
        try:
            names=['live_status','live_error','live_frames','live_pipeline_state','live_pipeline_error',
                'live_assets_ready','translator_lcd_errors','ocr_error']
            def snapshot():
                return {name:link.memory_read32(symbols[name],1)[0] for name in names if name in symbols}
            first=snapshot()
            deadline=time.monotonic()+20
            while first.get('live_pipeline_state')==3 and time.monotonic()<deadline:
                time.sleep(1);first=snapshot()
            time.sleep(2);second=snapshot()
            healthy=(second.get('live_assets_ready')==1 and second.get('live_error',0)==0
                and second.get('live_pipeline_state')==4 and second.get('live_pipeline_error',0)==0)
            report={'restored':healthy,'elf':str(elf),'start_script_returncode':result.returncode,
                'first':first,'second':second,'frames_advanced':second.get('live_frames',0)>first.get('live_frames',0)}
            (ROOT/'restoration_report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2),flush=True)
            if not healthy:
                raise RuntimeError('Original firmware restored but app status is not healthy')
        finally:
            link.close()


if __name__=='__main__':main()
