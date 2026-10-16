from pathlib import Path
def main():
  import subprocess,os,json,hashlib,struct
  import argparse
  p=argparse.ArgumentParser(description='Compare behavioral DRAM storage representations using all memory ports.')
  p.add_argument('--reference-root',type=Path,required=True)
  p.add_argument('--candidate-root',type=Path,required=True)
  p.add_argument('--verilator',type=Path,required=True)
  p.add_argument('--verilator-root',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  a=p.parse_args()
  out=a.output.resolve()
  out.mkdir(exist_ok=True,parents=True)
  env={**os.environ,'VERILATOR_ROOT':str(a.verilator_root.resolve())}
  verilator=a.verilator.resolve()
  image=b''.join(struct.pack('>I',((i*0x9e3779b9)^0x12345678)&0xffffffff) for i in range(65540))
  expected=bytearray(131072*4)
  for i in range(65540): struct.pack_into('<I',expected,4*i,((i*0x9e3779b9)^0x12345678)&0xffffffff)
  for addr,data,mask in [(65536,0xaabbccdd,5),(65535,0x11223344,10)]:
   for b in range(4):
    if mask>>b&1: expected[4*addr+b]=(data>>(8*b))&255
  for w in range(32):
   if w%3==0: struct.pack_into('<I',expected,4*(65520+w),0x10203040+w)
  results={}
  for name,tree in [('original',a.reference_root),('banked',a.candidate_root)]:
   d=out/name;d.mkdir(exist_ok=True);(d/'dram_0.bin').write_bytes(image)
   cmd=[str(verilator),'--binary','-j','4','-Wno-fatal','-O3','--x-assign','0','--x-initial','0','--top-module','tb_dram','--Mdir',str(d/'build'),str(tree.resolve()/'rtl/mem/otpu_dram.sv'),str(Path(__file__).with_name('validation')/'qwen17/dram_storage_tb.sv')]
   c=subprocess.run(cmd,env=env,capture_output=True,text=True)
   (d/'compile.log').write_text(c.stdout+c.stderr);c.check_returncode()
   r=subprocess.run([str(d/'build/Vtb_dram'),'+dir='+str(d)],check=True,capture_output=True,text=True,env=env)
   (d/'run.log').write_text(r.stdout+r.stderr)
   data=(d/'dram_out_0.bin').read_bytes()
   assert data==expected,(name,'output memory differs')
   results[name]={'read_trace':[s for s in r.stdout.splitlines() if s.startswith('READ ')],
   'output_sha256':hashlib.sha256(data).hexdigest(),'rtl_source_sha256':hashlib.sha256((tree.resolve()/'rtl/mem/otpu_dram.sv').read_bytes()).hexdigest()}
  assert results['original']['read_trace']==results['banked']['read_trace']
  assert len(results['banked']['read_trace'])==6
  report={'result':'pass','input_words':65540,'memory_words':131072,'tested':['port A aligned reads and byte writes','port SW byte writes','port B reads/writes crossing 65536-word storage boundary','request-response cycles and tags','partial final input row and zero-filled remaining memory','complete output dump versus independent expected bytes'],**results}
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__ == '__main__':
  main()
