"""Read-only diagnostic: no halt/reset, three samples of MCU/NPU progress."""
import fcntl
import json
import subprocess
import time
from run_board import ROOT, ELF, TOOLS, LIB, pylink


def main():
    with open('/tmp/alif-e8-1219599493.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        symbols={line.split()[2]:int(line.split()[0],16)
            for line in subprocess.check_output([str(TOOLS/'arm-none-eabi-nm'),'-n',str(ELF)],text=True).splitlines()
            if len(line.split())==3 and line.split()[0]!='U'}
        link=pylink.JLink(lib=pylink.library.Library(dllpath=LIB))
        link.open(serial_no=1219599493)
        link.set_tif(pylink.enums.JLinkInterfaces.SWD)
        link.connect('AE822FA0E5597LS0_M55_HP',speed=1000)
        samples=[]
        try:
            for i in range(3):
                words=link.memory_read32(symbols['act_benchmark'],32)
                snapshot=link.memory_read32(symbols['act_npu_snapshot'],163)
                npu=link.memory_read32(0x49042000,160)
                ticks=link.memory_read32(symbols['act_ticks'],1)[0]
                sample={'words':words,'npu_snapshot':snapshot,'npu_registers':npu,'ticks':ticks,
                    'npu_elapsed_ms':ticks-snapshot[0],'npu_status':npu[1],'qread':npu[6],'qsize':npu[8]}
                samples.append(sample)
                print({k:sample[k] for k in ['ticks','npu_elapsed_ms','npu_status','qread','qsize']},'state/run/phase',[words[1],words[6],words[7]],flush=True)
                if i<2:time.sleep(5)
            report={'status':'read_only_progress_diagnostic','samples':samples,
                'completed_prediction_runs':sum(x>0 for x in words[26:32]),
                'mcu_timer_advanced':samples[-1]['ticks']>samples[0]['ticks'],
                'qread_advanced':len({s['qread'] for s in samples})>1,
                'scope':'Exact two-camera 640x480 ACT checkpoint, INT8. Hardware diagnostic, no task-quality evaluation.'}
            (ROOT/'board_progress_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
        finally:
            link.close()


if __name__=='__main__':main()
