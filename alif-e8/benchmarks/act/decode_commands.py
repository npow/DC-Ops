"""Decode preserved U85 command bytes without accessing the board."""
import json,re,struct,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
pack_root=Path(os.environ.get('CMSIS_PACK_ROOT',Path.home()/'.cache/arm/packs'))
header=(pack_root/'ARM/ethos-u-core-driver/1.26.5/ethos_u_core_driver/src/ethosu85_interface.h').read_text()
tables={}
for control in [0,1]:
 body=re.search(r'enum class cmd'+str(control)+r'_opcode[^\{]*\{(.*?)\};',header,re.S).group(1)
 tables[control]={int(v,0):n for n,v in re.findall(r'(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+)',body)}
for name in ['vision','head']:
 raw=np.load(ROOT/f'{name}_vela/output/out_vela.npz',allow_pickle=True)['cmd_data'].tobytes()[32:]
 rows=[];offset=0;state={}
 while offset+4<=len(raw):
  word=struct.unpack_from('<I',raw,offset)[0];control=(word>>14)&3;op=word&0x3fff
  value=struct.unpack_from('<I',raw,offset+4)[0] if control==1 else word>>16
  label=tables.get(control,{}).get(op,f'unknown_{control}_{op}')
  row={'offset':offset,'op':label,'value':value,'hex':hex(value)}
  state[label]=value
  if label in ['NPU_OP_CONV','NPU_OP_DEPTHWISE','NPU_OP_POOL','NPU_OP_ELEMENTWISE']:
   row['dimensions']={k:v for k,v in state.items() if any(s in k for s in ['PRECISION','HEIGHT','WIDTH','DEPTH','KERNEL'])}
   print(name,offset,label,value,row['dimensions'])
  rows.append(row);offset+=8 if control==1 else 4
 (ROOT/f'{name}_commands.json').write_text(json.dumps(rows,indent=2)+'\n')
