"""Inspect lossless storage opportunities in actual retained MM weight bytes.

This is a data study, not implemented hardware or an LPN speedup. It preserves
the original logical input image and uses no neural-network computation.
"""
import argparse
import heapq
import json
from pathlib import Path
import time
import zlib


def merge_ranges(ranges):
  result=[]
  for lo,hi in sorted(ranges):
    if result and lo<=result[-1][1]:result[-1]=(result[-1][0],max(hi,result[-1][1]))
    else:result.append((lo,hi))
  return result


def immutable_blocks(ranges,writes,block=64):
  """Exclude every block intersecting a possible write, including partial words."""
  writes=merge_ranges(writes);result=[];j=0
  for lo,hi in merge_ranges(ranges):
    cursor=lo
    while j<len(writes) and writes[j][1]<=lo:j+=1
    k=j
    while k<len(writes) and writes[k][0]<hi:
      end=min(hi,writes[k][0]);begin=(cursor+block-1)//block*block;end=end//block*block
      if begin<end:result.append((begin,end))
      cursor=max(cursor,writes[k][1]);k+=1
    begin=(cursor+block-1)//block*block;end=hi//block*block
    if begin<end:result.append((begin,end))
  return result


def storage_statistics(data):
  import numpy as np
  data=np.asarray(data,dtype=np.uint8)
  if not len(data):raise ValueError('no immutable weight blocks to study')
  histogram=np.bincount(data,minlength=256)
  counts=histogram[histogram>0];probability=counts/data.size
  entropy=float(-(probability*np.log2(probability)).sum())
  heap=[int(c) for c in counts];heapq.heapify(heap);weighted_bits=0
  while len(heap)>1:
    total=heapq.heappop(heap)+heapq.heappop(heap);weighted_bits+=total;heapq.heappush(heap,total)
  # A 64-byte independent block allows bounded parallel hardware decoders.
  if len(data)%64:raise ValueError('whole aligned weight blocks required')
  signed=data.view(np.int8).astype(np.int16).reshape(-1,64)
  zigzag=np.where(signed>=0,2*signed,-2*signed-1).astype(np.uint16)
  lengths=np.stack([64*(k+1)+np.right_shift(zigzag,k).sum(axis=1) for k in range(8)])
  # Four bits per block: k=0..7 or escape to the original 512 data bits.
  rice_bits=int((np.minimum(lengths.min(axis=0),512)+4).sum())
  raw=data.tobytes();sizes={};verified={}
  for block in (4096,65536):
    size=0
    for start in range(0,len(raw),block):
      payload=raw[start:start+block];encoded=zlib.compress(payload,level=6)
      if zlib.decompress(encoded)!=payload:raise RuntimeError('lossless round trip failed')
      size+=min(len(encoded),len(payload))+4 # block descriptor, including escape
    sizes[str(block)]=size;verified[str(block)]=True
  return dict(raw_bytes=len(data),byte_entropy_bits=entropy,
      ideal_order0_bytes=entropy*len(data)/8,global_huffman_payload_bytes=(weighted_bits+7)//8,
      huffman_scope='Payload only: no codebook, indexing, alignment or decoder cost',
      rice64_payload_and_mode_bytes=(rice_bits+7)//8,
      rice64_scope='Size estimate only: no row directory or hardware throughput/area proof',
      zlib_block_bytes=sizes,zlib_lossless_round_trip_verified=verified,
      zlib_scope='Software diagnostic with block descriptors; no FPGA decoder implementation',
      byte_histogram=histogram.tolist())


def read_image(path,size):
  import numpy as np
  path=Path(path)
  if path.stat().st_size%4 or path.stat().st_size>size:raise ValueError('invalid retained image size')
  prefix=np.fromfile(path,dtype='>u4').astype('<u4').view(np.uint8)
  image=np.zeros(size,dtype=np.uint8);image[:len(prefix)]=prefix
  return image


def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('case','opentpu-root','output'):p.add_argument('--'+name,type=Path,required=True)
  p.add_argument('--kernel',type=int,default=16)
  a=p.parse_args()
  import numpy as np
  from .e2e import atomic_json,digest
  from .workload import use_checkout
  from .retained_inputs import verified_program
  from .shape_workload import resolve_static
  use_checkout(a.opentpu_root.resolve())
  from opentpu import Config,isa as I,waits as W
  reference=json.loads((a.case/'report.json').read_text())
  if reference['status']!='complete' or not reference['all_dram_and_tmem_bit_exact']:
    raise ValueError('completed verified inference required')
  cfg=Config(**reference['config'])
  if cfg.S!=1:raise ValueError('study currently requires one slice')
  source=a.case/'kernels'/f'{a.kernel:04d}';record=reference['kernels'][a.kernel]
  for name in ('prog_0.hex','dram_0.bin'):
    if digest(source/name)!=record['input_sha256'][name]:raise ValueError('retained input changed')
  encoded,identity=verified_program(source,0)
  work=resolve_static(cfg,[[I.Instr.decode(words) for words in encoded]],record['args'])
  ranges=set();requested=0;writes=[]
  for c in work.commands[0]:
    writes.extend((lo,hi) for space,lo,hi in c.footprint.wr if space==W.SP_DRAM)
    ins=c.instruction
    if ins.op!=I.MM:continue
    if ((ins.flags>>I.WF_SHIFT)&3)!=I.W8:raise ValueError('current codec study requires int8 weights')
    w=ins.w;n,kb=w[3]&65535,(w[3]>>16)&255;size=kb*cfg.D;requested+=n*size
    ranges.update((w[0]+row*w[4],w[0]+row*w[4]+size) for row in range(n))
  unique=merge_ranges(ranges);writes=merge_ranges(writes)
  # Restrict compression candidates to full ranges untouched by kernel writes.
  # A separate cross-kernel coherence check is required for persistent storage.
  immutable=immutable_blocks(unique,writes)
  image=read_image(source/'dram_0.bin',cfg.DRAM_BYTES)
  data=np.concatenate([image[lo:hi] for lo,hi in immutable])
  before=time.perf_counter();stats=storage_statistics(data)
  result=dict(status='complete',scope=__doc__,kernel=a.kernel,input_sha256=record['input_sha256'],
      program_json_sha256=identity,source_sha256=digest(Path(__file__)),
      requested_weight_bytes=requested,unique_weight_bytes=sum(hi-lo for lo,hi in unique),
      kernel_immutable_weight_bytes=len(data),immutable_ranges=immutable,statistics=stats,
      analysis_seconds=time.perf_counter()-before,numerical_arithmetic_executed=False,
      hardware_implemented=False,limitations=['One kernel, not full inference',
          'No timing or speedup claim','Kernel-local immutability does not prove cross-kernel coherence',
          'A physical codec needs bounded decode throughput, metadata and resource validation'])
  atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('statistics','immutable_ranges')},indent=2));print(json.dumps({k:v for k,v in stats.items() if k!='byte_histogram'},indent=2))


if __name__=='__main__':main()
