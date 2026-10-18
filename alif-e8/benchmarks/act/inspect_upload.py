import fcntl,hashlib,json,subprocess
from pathlib import Path
from run_board import ELF,TOOLS,ROOT,LIB,pylink
with open('/tmp/alif-e8-1219599493.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 symbols={line.split()[2]:int(line.split()[0],16) for line in subprocess.check_output([str(TOOLS/'arm-none-eabi-nm'),'-n',str(ELF)],text=True).splitlines() if len(line.split())==3 and line.split()[0]!='U'}
 link=pylink.JLink(lib=pylink.library.Library(dllpath=LIB));link.open(serial_no=1219599493);link.set_tif(pylink.enums.JLinkInterfaces.SWD);link.connect('AE822FA0E5597LS0_M55_HP',speed=1000)
 try:
  raw=(ROOT/'vision.pte').read_bytes()[:65536]
  report={}
  for name,address in [('staging',symbols['temp_pool']),('psram',0xa0000000)]:
   actual=bytes(link.memory_read8(address,len(raw)));bad=[i for i,(a,b) in enumerate(zip(raw,actual)) if a!=b]
   report[name]={'address':hex(address),'bad':len(bad),'even_bad':sum(i%2==0 for i in bad),'first_bad':bad[:16],'sha256':hashlib.sha256(actual).hexdigest()}
  report['mailbox']=link.memory_read32(symbols['act_benchmark'],8)
  (ROOT/'fixed_latency_upload_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 finally:link.close()
